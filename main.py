from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import CONFIG
from utils import detect_smiles_column, labels_to_int, stratified_scaffold_split


def _log_torch_runtime(task_name: str) -> None:
    print(f"[{task_name}] Torch runtime diagnostics")
    print(f"  torch_version: {torch.__version__}")
    print(f"  torch_cuda_available: {torch.cuda.is_available()}")
    print(f"  torch_cuda_version: {torch.version.cuda}")
    print(f"  cuda_device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_id)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  active_device: cuda:{dev_id} - {props.name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("  active_device: CPU")
        print("  WARNING: CUDA not available in current PyTorch build.")
        print("  Hint: install CUDA-enabled PyTorch in this env (not +cpu build).")


def _load_and_split() -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    df = pd.read_csv(CONFIG["DATA_CSV"])
    smiles_col = detect_smiles_column(df)
    label_col = CONFIG["LABEL_COLUMN"]
    train_df, valid_df = stratified_scaffold_split(
        df=df,
        smiles_col=smiles_col,
        label_col=label_col,
        test_size=CONFIG["TEST_SIZE"],
        seed=CONFIG["RANDOM_STATE"],
    )
    split_dir = Path(CONFIG["SPLIT_DIR"])
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train_split.csv", index=False)
    valid_df.to_csv(split_dir / "valid_split.csv", index=False)
    return train_df, valid_df, smiles_col, label_col


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], labels: list[int], tokenizer, max_len: int) -> None:
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_finetuning() -> None:
    _log_torch_runtime("finetuning")
    train_df, valid_df, smiles_col, label_col = _load_and_split()
    y_train = labels_to_int(train_df[label_col])
    y_valid = labels_to_int(valid_df[label_col])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_PATH"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["MODEL_PATH"], num_labels=2)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    model.to(device)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  acceleration: CUDA + AMP(fp16) + TF32 enabled")
    else:
        print("  acceleration: CPU only")

    train_ds = SmilesDataset(train_df[smiles_col].astype(str).tolist(), y_train.tolist(), tokenizer, CONFIG["BERT_MAX_LENGTH"])
    valid_ds = SmilesDataset(valid_df[smiles_col].astype(str).tolist(), y_valid.tolist(), tokenizer, CONFIG["BERT_MAX_LENGTH"])
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["BERT_BATCH_SIZE"],
        shuffle=True,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CONFIG["BERT_BATCH_SIZE"],
        shuffle=False,
        pin_memory=use_cuda,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["FINETUNE_LR"],
        weight_decay=CONFIG["FINETUNE_WEIGHT_DECAY"],
    )

    best_f1 = -1.0
    best_state = None
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    for epoch in range(CONFIG["FINETUNE_EPOCHS"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Finetune epoch {epoch + 1}/{CONFIG['FINETUNE_EPOCHS']}", unit="batch")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in valid_loader:
                labels = batch["labels"].numpy()
                batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(**batch).logits
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                refs.extend(labels.tolist())
        f1 = f1_score(refs, preds, average="macro")
        print(f"Validation F1-macro epoch {epoch + 1}: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    finetuned_dir = Path(CONFIG["FINETUNED_DIR"])
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(finetuned_dir)
    tokenizer.save_pretrained(finetuned_dir)

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Predict MolDeBERTa on valid", unit="batch"):
            batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(**batch).logits
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob.tolist())

    pred_dir = Path(CONFIG["PRED_DIR"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "smiles": valid_df[smiles_col].astype(str).tolist(),
            "label": y_valid.tolist(),
            "mol_score": probs,
        }
    ).to_csv(pred_dir / "mol_predictions.csv", index=False)
    print(f"Saved finetuned model: {finetuned_dir}")
    print(f"Saved MolDeBERTa predictions: {pred_dir / 'mol_predictions.csv'}")


def run_buildkg() -> None:
    _ensure_neo4j_ready()
    from kg.build_graph import KnowledgeGraphBuilder

    train_df, _, smiles_col, label_col = _load_and_split()
    with KnowledgeGraphBuilder(config_path=CONFIG["DOMAIN_CONFIG_PATH"]) as builder:
        builder.nuke_and_prepare_db()
        builder.process_experimental_molecules(train_df, smiles_col=smiles_col, label_col=label_col)
        denovo_path = Path(CONFIG["DENOVO_CSV"])
        if denovo_path.exists():
            denovo_df = pd.read_csv(denovo_path)
            if not denovo_df.empty:
                builder.process_denovo_molecules(denovo_df)
    print("buildkg completed: train molecules + de novo molecules imported to Neo4j.")


def run_kg_training() -> None:
    _ensure_neo4j_ready()
    from kg.kg_encoder import build_train_and_valid_kg_features

    train_df, valid_df, smiles_col, label_col = _load_and_split()
    y_train = labels_to_int(train_df[label_col])
    y_valid = labels_to_int(valid_df[label_col])

    print("Extracting train KG features + inferring valid features via Top-K Tanimoto neighbors...")
    kg_train, kg_valid = build_train_and_valid_kg_features(
        train_smiles=train_df[smiles_col].astype(str).tolist(),
        valid_smiles=valid_df[smiles_col].astype(str).tolist(),
        target_name="EGFR",
        top_k=5,
    )
    print("KG feature diagnostics:")
    print(f"  Train shape: {kg_train.shape}, Valid shape: {kg_valid.shape}")
    print(f"  Train zero rows: {(kg_train.sum(axis=1) == 0).mean():.2%}")
    print(f"  Valid zero rows: {(kg_valid.sum(axis=1) == 0).mean():.2%}")
    print(f"  Train feature mean: {kg_train.mean(axis=0)}")
    print(f"  Valid feature mean: {kg_valid.mean(axis=0)}")
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        eval_metric="logloss",
        random_state=CONFIG["RANDOM_STATE"],
    )
    clf.fit(kg_train, y_train)
    kg_probs = clf.predict_proba(kg_valid)[:, 1]
    print(f"KG score distribution -> min: {kg_probs.min():.6f}, max: {kg_probs.max():.6f}, std: {kg_probs.std():.6f}")

    pred_dir = Path(CONFIG["PRED_DIR"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "smiles": valid_df[smiles_col].astype(str).tolist(),
            "label": y_valid.tolist(),
            "kg_score": kg_probs,
        }
    ).to_csv(pred_dir / "kg_predictions.csv", index=False)
    pd.DataFrame(
        kg_train,
        columns=[
            "num_warheads",
            "num_moas",
            "has_egfr_path",
            "deg_total",
            "num_scaffolds",
            "num_targets",
            "num_interaction_groups",
            "num_functional_groups",
            "knn_mean_sim",
            "knn_max_sim",
            "knn_std_sim",
        ],
    ).assign(smiles=train_df[smiles_col].astype(str).tolist(), label=y_train).to_csv(pred_dir / "kg_features_train.csv", index=False)
    pd.DataFrame(
        kg_valid,
        columns=[
            "num_warheads",
            "num_moas",
            "has_egfr_path",
            "deg_total",
            "num_scaffolds",
            "num_targets",
            "num_interaction_groups",
            "num_functional_groups",
            "knn_mean_sim",
            "knn_max_sim",
            "knn_std_sim",
        ],
    ).assign(smiles=valid_df[smiles_col].astype(str).tolist(), label=y_valid).to_csv(pred_dir / "kg_features_valid_inferred.csv", index=False)
    print(f"Saved KG predictions: {pred_dir / 'kg_predictions.csv'}")


def run_molformer_training() -> None:
    _log_torch_runtime("molformer_training")
    train_df, valid_df, smiles_col, label_col = _load_and_split()
    y_train = labels_to_int(train_df[label_col])
    y_valid = labels_to_int(valid_df[label_col])

    model_name = CONFIG["MOLFORMER_MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )
    except TypeError:
        # Some transformer versions/models do not accept deterministic_eval.
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.float().eval()
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    model.to(device)

    def _encode(smiles_list: list[str], desc: str) -> np.ndarray:
        out = []
        bs = CONFIG["BERT_BATCH_SIZE"]
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), bs), total=(len(smiles_list) + bs - 1) // bs, desc=desc, unit="batch"):
                batch = smiles_list[i : i + bs]
                toks = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["BERT_MAX_LENGTH"],
                    return_tensors="pt",
                )
                input_ids = toks["input_ids"].to(device, non_blocking=use_cuda)
                attention_mask = toks["attention_mask"].to(device, non_blocking=use_cuda)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                h = outputs.last_hidden_state.float()

                # Stable pooled representations: CLS + masked mean pooling.
                cls = h[:, 0, :]
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                rep = torch.cat([cls, pooled], dim=1).detach().cpu().numpy().astype(np.float32)

                if np.isnan(rep).any() or np.isinf(rep).any():
                    np.nan_to_num(rep, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                out.append(rep)
        return np.vstack(out)

    train_emb = _encode(train_df[smiles_col].astype(str).tolist(), "MolFormer train embeddings")
    valid_emb = _encode(valid_df[smiles_col].astype(str).tolist(), "MolFormer valid embeddings")

    # Sanitize numeric issues from encoder outputs (NaN/Inf) before sklearn fit.
    n_bad_train = int((~np.isfinite(train_emb)).any(axis=1).sum())
    n_bad_valid = int((~np.isfinite(valid_emb)).any(axis=1).sum())
    if n_bad_train > 0 or n_bad_valid > 0:
        print(f"MolFormer embedding cleanup: bad_train_rows={n_bad_train}, bad_valid_rows={n_bad_valid}")

    # Replace non-finite values with per-column median (computed on finite train values).
    train_emb = train_emb.astype(np.float32, copy=False)
    valid_emb = valid_emb.astype(np.float32, copy=False)
    finite_train = np.where(np.isfinite(train_emb), train_emb, np.nan)
    col_med = np.nanmedian(finite_train, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    train_emb = np.where(np.isfinite(train_emb), train_emb, col_med[None, :])
    valid_emb = np.where(np.isfinite(valid_emb), valid_emb, col_med[None, :])

    # Drop near-constant embedding dimensions to avoid degenerate linear model.
    var = train_emb.var(axis=0)
    keep = var > 1e-10
    if not np.any(keep):
        print("MolFormer embeddings collapsed (all near-constant). Falling back to all dims.")
        keep = np.ones_like(var, dtype=bool)
    train_emb = train_emb[:, keep]
    valid_emb = valid_emb[:, keep]
    print(f"MolFormer embedding dims kept: {train_emb.shape[1]}/{len(var)}")
    print(f"MolFormer train embedding variance mean: {float(train_emb.var(axis=0).mean()):.6e}")
    print(f"MolFormer train unique rows ratio: {np.unique(train_emb.round(6), axis=0).shape[0] / max(1, train_emb.shape[0]):.3f}")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_emb)
    valid_x = scaler.transform(valid_emb)

    clf = LogisticRegression(
        max_iter=5000,
        random_state=CONFIG["RANDOM_STATE"],
        class_weight="balanced",
        solver="saga",
    )
    clf.fit(train_x, y_train)
    molformer_score = clf.predict_proba(valid_x)[:, 1]

    # If score collapses to near-constant, fallback to XGBoost on same embeddings.
    if float(np.std(molformer_score)) < 1e-5:
        print("MolFormer LR scores nearly constant, fallback to XGBoost classifier.")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            eval_metric="logloss",
            random_state=CONFIG["RANDOM_STATE"],
        )
        xgb.fit(train_x, y_train)
        molformer_score = xgb.predict_proba(valid_x)[:, 1]

    # If still collapsed, fallback to RandomForest for robust non-linear decision boundary.
    if float(np.std(molformer_score)) < 1e-5:
        print("MolFormer XGBoost scores still nearly constant, fallback to RandomForest.")
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=CONFIG["RANDOM_STATE"],
            n_jobs=-1,
        )
        rf.fit(train_x, y_train)
        molformer_score = rf.predict_proba(valid_x)[:, 1]

    print(
        "MolFormer score distribution -> "
        f"min: {molformer_score.min():.6f}, max: {molformer_score.max():.6f}, std: {molformer_score.std():.6f}"
    )

    pred_dir = Path(CONFIG["PRED_DIR"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "smiles": valid_df[smiles_col].astype(str).tolist(),
            "label": y_valid.tolist(),
            "molformer_score": molformer_score,
        }
    ).to_csv(pred_dir / "molformer_predictions.csv", index=False)
    print(f"Saved MolFormer predictions: {pred_dir / 'molformer_predictions.csv'}")


def run_benchmark() -> None:
    from evaluation.benchmark import run_benchmark

    pred_dir = Path(CONFIG["PRED_DIR"])
    molformer_file = pred_dir / "molformer_predictions.csv"
    if not molformer_file.exists():
        print("MolFormer predictions missing. Auto-running molformer_training before benchmark...")
        run_molformer_training()

    run_benchmark()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["buildkg", "finetuning", "molformer_training", "kg_training", "benchmark"], required=True)
    args = parser.parse_args()

    if args.mode == "buildkg":
        run_buildkg()
    elif args.mode == "finetuning":
        run_finetuning()
    elif args.mode == "molformer_training":
        run_molformer_training()
    elif args.mode == "kg_training":
        run_kg_training()
    elif args.mode == "benchmark":
        run_benchmark()


def _try_start_neo4j_docker() -> None:
    compose_file = Path(CONFIG["PROJECT_ROOT"]) / "docker-compose.yml"
    if not compose_file.exists():
        return
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "neo4j"],
            cwd=str(compose_file.parent),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def _ensure_neo4j_ready(max_wait_sec: int = 45) -> None:
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency `neo4j`. Install it in your active env with `pip install neo4j`."
        ) from exc

    uri = CONFIG["NEO4J_URI"]
    user = CONFIG["NEO4J_USER"]
    password = CONFIG["NEO4J_PASSWORD"]

    _try_start_neo4j_docker()
    deadline = time.time() + max_wait_sec
    last_error = None

    while time.time() < deadline:
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            print(f"Neo4j is ready at {uri}")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(3)
        finally:
            if driver is not None:
                driver.close()

    raise RuntimeError(
        "Cannot connect to Neo4j. "
        f"Current config: NEO4J_URI={uri}, NEO4J_USER={user}. "
        "Please start container `docker compose up -d neo4j` and verify password."
    ) from last_error


if __name__ == "__main__":
    main()
