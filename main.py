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
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
import inspect
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from rdkit import Chem

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


def _enumerate_smiles(smiles: str, n_random: int = 2, max_tries: int = 50) -> list[str]:
    """
    Generate up to `n_random` additional randomized SMILES (RDKit doRandom=True).
    Falls back to canonical SMILES duplicates if enumeration fails.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    base = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    out: list[str] = []
    seen = {base}
    tries = 0
    while len(out) < n_random and tries < max_tries:
        tries += 1
        try:
            s = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=False)
        except Exception:
            continue
        if not s or s in seen:
            continue
        # Validate generated SMILES round-trip.
        if Chem.MolFromSmiles(s) is None:
            continue
        seen.add(s)
        out.append(s)
    # If RDKit couldn't produce enough random SMILES, duplicate canonical ones to keep 3x training size.
    while len(out) < n_random:
        out.append(base)
    return out


class FocalLossTrainer(Trainer):
    """
    HuggingFace Trainer with auto-adaptive focal loss for binary classification.
    Dynamic alpha is computed per-batch from label counts:
      alpha_pos = n_neg / n_total, alpha_neg = n_pos / n_total
    """

    def __init__(self, *args, gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None or logits is None:
            raise RuntimeError("FocalLossTrainer requires `labels` and model `logits`.")

        labels = labels.view(-1).to(logits.device)
        logits = logits.view(-1, logits.shape[-1])

        # Compute per-batch class weights without peeking at validation/test.
        n_total = max(int(labels.numel()), 1)
        n_pos = int((labels == 1).sum().item())
        n_neg = n_total - n_pos
        alpha_pos = float(n_neg) / float(n_total)  # weight for positive samples
        alpha_neg = float(n_pos) / float(n_total)  # weight for negative samples

        logp = torch.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        alpha_t = torch.where(labels == 1, torch.tensor(alpha_pos, device=logits.device), torch.tensor(alpha_neg, device=logits.device))
        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * logpt
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def run_finetuning() -> None:
    _log_torch_runtime("finetuning")
    train_df, valid_df, smiles_col, label_col = _load_and_split()
    y_train = labels_to_int(train_df[label_col])
    y_valid = labels_to_int(valid_df[label_col])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_PATH"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["MODEL_PATH"], num_labels=2)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  acceleration: CUDA + AMP(fp16) + TF32 enabled")
    else:
        print("  acceleration: CPU only")

    # -----------------------
    # Step 1) Train-only SMILES enumeration augmentation (triples train size).
    # -----------------------
    train_base_n = int(len(train_df))
    aug_rows = []
    for s, y in zip(train_df[smiles_col].astype(str).tolist(), y_train.tolist()):
        for rs in _enumerate_smiles(s, n_random=2):
            aug_rows.append({smiles_col: rs, label_col: y})
    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        # Ensure labels are represented consistently for downstream mapping.
        aug_df[label_col] = aug_df[label_col].map(lambda v: "active" if int(v) == 1 else "inactive")
        train_df_aug = pd.concat([train_df[[smiles_col, label_col]].copy(), aug_df], ignore_index=True)
    else:
        train_df_aug = train_df[[smiles_col, label_col]].copy()

    y_train_aug = labels_to_int(train_df_aug[label_col])
    print(
        "Train augmentation (SMILES enumeration): "
        f"original_train={train_base_n}, augmented_train={len(train_df_aug)} (target ~{train_base_n * 3})"
    )
    n_pos = int((y_train_aug == 1).sum())
    n_neg = int((y_train_aug == 0).sum())
    alpha_pos = n_neg / max(n_pos + n_neg, 1)
    alpha_neg = n_pos / max(n_pos + n_neg, 1)
    print(f"FocalLoss dynamic alpha (from augmented train): alpha_pos={alpha_pos:.4f}, alpha_neg={alpha_neg:.4f}, gamma=2.0")

    # HuggingFace datasets + tokenizer
    train_hf = HFDataset.from_pandas(pd.DataFrame({"smiles": train_df_aug[smiles_col].astype(str), "labels": y_train_aug}))
    valid_hf = HFDataset.from_pandas(pd.DataFrame({"smiles": valid_df[smiles_col].astype(str), "labels": y_valid}))

    def _tok(batch):
        return tokenizer(
            batch["smiles"],
            truncation=True,
            padding="max_length",
            max_length=CONFIG["BERT_MAX_LENGTH"],
        )

    train_hf = train_hf.map(_tok, batched=True, remove_columns=["smiles"])
    valid_hf = valid_hf.map(_tok, batched=True, remove_columns=["smiles"])
    train_hf.set_format(type="torch")
    valid_hf.set_format(type="torch")

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"f1_macro": float(f1_score(labels, preds, average="macro"))}

    finetuned_dir = Path(CONFIG["FINETUNED_DIR"])
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    # transformers version compatibility: only pass supported TrainingArguments kwargs
    ta_sig = inspect.signature(TrainingArguments.__init__)
    supported = set(ta_sig.parameters.keys())
    args_kwargs = {
        "output_dir": str(finetuned_dir / "trainer_runs"),
        "num_train_epochs": CONFIG["FINETUNE_EPOCHS"],
        "per_device_train_batch_size": CONFIG["BERT_BATCH_SIZE"],
        "per_device_eval_batch_size": CONFIG["BERT_BATCH_SIZE"],
        "learning_rate": CONFIG["FINETUNE_LR"],
        "weight_decay": CONFIG["FINETUNE_WEIGHT_DECAY"],
        "fp16": use_cuda,
        "logging_steps": 50,
        "seed": CONFIG["RANDOM_STATE"],
        # Newer transformers
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "report_to": [],
        # Older transformers fallbacks
        "do_eval": True,
        "evaluate_during_training": True,
    }
    filtered = {k: v for k, v in args_kwargs.items() if k in supported}
    if "evaluation_strategy" not in filtered and "do_eval" in supported:
        filtered["do_eval"] = True

    # Compatibility guard: some transformers versions require eval/save strategies to match
    # when load_best_model_at_end=True. If we can't set evaluation strategy, disable it.
    if filtered.get("load_best_model_at_end") is True:
        eval_strat = filtered.get("evaluation_strategy", None)
        save_strat = filtered.get("save_strategy", None)
        if eval_strat is None or save_strat is None or eval_strat != save_strat:
            print(
                "TrainingArguments compatibility: disabling load_best_model_at_end "
                f"(evaluation_strategy={eval_strat}, save_strategy={save_strat})"
            )
            filtered["load_best_model_at_end"] = False
            # These are only meaningful with best-model loading.
            filtered.pop("metric_for_best_model", None)
            filtered.pop("greater_is_better", None)

    args = TrainingArguments(**filtered)

    trainer = FocalLossTrainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=valid_hf,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        gamma=2.0,
    )
    trainer.train()

    # Save best model + tokenizer.
    trainer.model.save_pretrained(finetuned_dir)
    tokenizer.save_pretrained(finetuned_dir)

    # Predict probabilities on VALID only (no augmentation, no leakage).
    pred_out = trainer.predict(valid_hf)
    valid_logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(valid_logits), dim=1)[:, 1].cpu().numpy().tolist()

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
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg) / float(max(n_pos, 1))
    print(f"XGBoost dynamic scale_pos_weight (from train split): neg={n_neg}, pos={n_pos}, scale_pos_weight={pos_weight:.6f}")

    print("Extracting train KG features + inferring valid features via Top-K Tanimoto neighbors...")
    kg_train, kg_valid = build_train_and_valid_kg_features(
        train_smiles=train_df[smiles_col].astype(str).tolist(),
        valid_smiles=valid_df[smiles_col].astype(str).tolist(),
        train_y=y_train,
        target_name="EGFR",
        top_k=5,
    )
    feature_cols = [
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
        "knn_pos_rate",
        "knn_pos_rate_w",
        "knn_nearest_pos_sim",
        "knn_nearest_neg_sim",
    ]

    print("KG feature diagnostics:")
    print(f"  Train shape: {kg_train.shape}, Valid shape: {kg_valid.shape}")
    print(f"  Train zero rows: {(kg_train.sum(axis=1) == 0).mean():.2%}")
    print(f"  Valid zero rows: {(kg_valid.sum(axis=1) == 0).mean():.2%}")
    print(f"  Train feature mean: {kg_train.mean(axis=0)}")
    print(f"  Valid feature mean: {kg_valid.mean(axis=0)}")

    # Remove useless columns (all-zero / near-constant on train split).
    train_var = kg_train.var(axis=0)
    keep_mask = train_var > 1e-10
    if not np.any(keep_mask):
        raise RuntimeError("All KG features are near-constant. Cannot train KG model.")
    dropped_cols = [c for c, k in zip(feature_cols, keep_mask) if not k]
    kept_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
    kg_train = kg_train[:, keep_mask]
    kg_valid = kg_valid[:, keep_mask]
    print(f"  Kept KG features ({len(kept_cols)}): {kept_cols}")
    if dropped_cols:
        print(f"  Dropped constant KG features ({len(dropped_cols)}): {dropped_cols}")

    # Early stopping split from TRAIN only (no validation leakage).
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=CONFIG["RANDOM_STATE"])
    tr_idx, es_idx = next(sss.split(kg_train, y_train))
    x_tr, y_tr = kg_train[tr_idx], y_train[tr_idx]
    x_es, y_es = kg_train[es_idx], y_train[es_idx]
    print(f"KG XGB early-stopping split (train-only): train={len(tr_idx)}, holdout={len(es_idx)}")

    clf = XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        gamma=0.0,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=CONFIG["RANDOM_STATE"],
    )
    clf.fit(
        x_tr,
        y_tr,
        eval_set=[(x_es, y_es)],
        verbose=False,
    )
    if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        print(f"KG XGB best_iteration: {int(clf.best_iteration)}")

    # Refit on full train split at best_iteration for stable final model.
    best_n = int(getattr(clf, "best_iteration", 0) or 0) + 1
    best_n = max(50, min(best_n, 2000))
    clf = XGBClassifier(
        n_estimators=best_n,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        gamma=0.0,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        eval_metric="logloss",
        random_state=CONFIG["RANDOM_STATE"],
    )
    clf.fit(kg_train, y_train)

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=np.float32)
        topk = int(min(10, imp.size))
        order = np.argsort(-imp)[:topk]
        print("Top KG feature importances:")
        for j in order.tolist():
            name = kept_cols[j] if j < len(kept_cols) else f"f{j}"
            print(f"  {name}: {float(imp[j]):.6f}")

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
        columns=kept_cols,
    ).assign(smiles=train_df[smiles_col].astype(str).tolist(), label=y_train).to_csv(pred_dir / "kg_features_train.csv", index=False)
    pd.DataFrame(
        kg_valid,
        columns=kept_cols,
    ).assign(smiles=valid_df[smiles_col].astype(str).tolist(), label=y_valid).to_csv(pred_dir / "kg_features_valid_inferred.csv", index=False)
    pd.DataFrame(
        {
            "kept_feature": kept_cols,
        }
    ).to_csv(pred_dir / "kg_selected_features.csv", index=False)
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
    parser.add_argument(
        "--mode",
        choices=["buildkg", "finetuning", "molformer_training", "kg_training", "benchmark", "all"],
        required=True,
    )
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
    elif args.mode == "all":
        print("Running full pipeline (all modes) in order:")
        steps = [
            ("buildkg", run_buildkg),
            ("molformer_training", run_molformer_training),
            ("finetuning", run_finetuning),
            ("kg_training", run_kg_training),
            ("benchmark", run_benchmark),
        ]
        for name, fn in steps:
            print(f"\n{'=' * 80}\nMODE: {name}\n{'=' * 80}")
            fn()


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
