from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import torch

from config import CONFIG
from evaluation.fingerprints import assign_bins, compute_fps, max_train_similarity
from evaluation.metrics import classification_metrics, fit_predict_proba, get_models
from evaluation.visualization_script import (
    plot_active_prediction_counts,
    plot_auc_comparison_detailed,
    plot_heatmap_detailed,
    plot_improvement_detailed,
    plot_model_metrics_compairison,
    plot_roc_comparison,
)
from utils import detect_smiles_column, stratified_scaffold_split


def _extract_molformer_embeddings(train_df: pd.DataFrame, valid_df: pd.DataFrame, smiles_col: str):
    model_name = CONFIG["MOLFORMER_MODEL_PATH"]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")

    def encode(smiles: list[str]) -> np.ndarray:
        outs = []
        batch_size = CONFIG["BERT_BATCH_SIZE"]
        total_batches = (len(smiles) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(smiles), batch_size), total=total_batches, desc="MolFormer embedding batches", unit="batch"):
            batch = smiles[i : i + CONFIG["BERT_BATCH_SIZE"]]
            inp = tok(batch, padding=True, truncation=True, max_length=CONFIG["BERT_MAX_LENGTH"], return_tensors="pt")
            if torch.cuda.is_available():
                inp = {k: v.to("cuda") for k, v in inp.items()}
            with torch.no_grad():
                h = mdl(**inp).last_hidden_state
                m = inp["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            outs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outs)

    return encode(train_df[smiles_col].astype(str).tolist()), encode(valid_df[smiles_col].astype(str).tolist())


def run_benchmark_from_embeddings(out_dir: Path) -> Dict:
    data_df = pd.read_csv(CONFIG["DATA_CSV"])
    smiles_col = detect_smiles_column(data_df)
    label_col = CONFIG["LABEL_COLUMN"]
    train_df, valid_df = stratified_scaffold_split(data_df, smiles_col, label_col, CONFIG["TEST_SIZE"], CONFIG["RANDOM_STATE"])
    y_train = train_df[label_col].astype(str).str.lower().map(lambda x: 1 if x in {"active", "1", "true", "yes"} else 0).to_numpy()
    y_valid = valid_df[label_col].astype(str).str.lower().map(lambda x: 1 if x in {"active", "1", "true", "yes"} else 0).to_numpy()

    bert_train = np.load(out_dir / "bert_train.npy")
    bert_valid = np.load(out_dir / "bert_valid.npy")
    kg_train = np.load(out_dir / "kg_train.npy")
    kg_valid = np.load(out_dir / "kg_valid.npy")
    bertkg_train = np.concatenate([bert_train, kg_train], axis=1)
    bertkg_valid = np.concatenate([bert_valid, kg_valid], axis=1)

    representations = {
        "MolDeBERTa": (bert_train, bert_valid),
        "MolDeBERTa x KG": (bertkg_train, bertkg_valid),
    }
    try:
        mf_train, mf_valid = _extract_molformer_embeddings(train_df, valid_df, smiles_col)
        representations["MolFormer"] = (mf_train, mf_valid)
    except Exception as exc:
        print(f"MolFormer benchmark skipped: {exc}")

    bins_cfg = CONFIG["TANIMOTO_BINS"]
    bin_names = [b[2] for b in bins_cfg]
    valid_bins = assign_bins(max_train_similarity(compute_fps(valid_df[smiles_col].astype(str).tolist()), compute_fps(train_df[smiles_col].astype(str).tolist())), bins_cfg)

    all_results: Dict = {}
    roc_data: Dict = {}
    active_counts: Dict = {}
    models = get_models()
    for method_name, (X_train, X_valid) in tqdm(representations.items(), total=len(representations), desc="Benchmark methods", unit="method"):
        method_overall_probs = []
        method_res = {"overall_by_model": {}, "by_bin": {}}
        for model_name, model in tqdm(models.items(), total=len(models), desc=f"{method_name} classifiers", unit="clf", leave=False):
            probs = fit_predict_proba(model, X_train, y_train, X_valid)
            method_overall_probs.append(probs)
            method_res["overall_by_model"][model_name] = classification_metrics(y_valid, probs)
            for bin_idx, (_, _, bin_name) in enumerate(bins_cfg):
                mask = valid_bins == bin_idx
                if mask.sum() < 2:
                    continue
                method_res["by_bin"].setdefault(bin_name, {})
                method_res["by_bin"][bin_name][model_name] = classification_metrics(y_valid[mask], probs[mask])
        mean_prob = np.mean(np.vstack(method_overall_probs), axis=0)
        method_res["overall"] = classification_metrics(y_valid, mean_prob)
        fpr, tpr, _ = roc_curve(y_valid, mean_prob)
        roc_data[method_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        active_counts[method_name] = {}
        for bin_idx, (_, _, bin_name) in enumerate(bins_cfg):
            mask = valid_bins == bin_idx
            active_counts[method_name][bin_name] = int(((mean_prob[mask] >= 0.5).astype(np.int64)).sum())
        all_results[method_name] = method_res

    eval_dir = Path(CONFIG["EVAL_DIR"])
    plot_dir = Path(CONFIG["PLOT_DIR"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_auc_comparison_detailed(all_results, bin_names, plot_dir)
    plot_active_prediction_counts(active_counts, bin_names, plot_dir)
    plot_heatmap_detailed(all_results, bin_names, plot_dir)
    if "MolDeBERTa x KG" in all_results:
        plot_improvement_detailed(all_results, bin_names, plot_dir)
    plot_model_metrics_compairison(all_results, plot_dir)
    plot_roc_comparison(roc_data, plot_dir)

    payload = {"results": all_results, "roc": roc_data, "active_counts": active_counts}
    (eval_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overall_rows = []
    by_bin_rows = []
    for method_name, method_data in all_results.items():
        overall = method_data.get("overall", {})
        overall_rows.append(
            {
                "method": method_name,
                "accuracy": overall.get("accuracy"),
                "precision": overall.get("precision"),
                "recall": overall.get("recall"),
                "f1": overall.get("f1"),
                "auc": overall.get("auc"),
            }
        )
        for bin_name, bin_data in method_data.get("by_bin", {}).items():
            for model_name, metrics in bin_data.items():
                by_bin_rows.append(
                    {
                        "method": method_name,
                        "bin": bin_name,
                        "classifier": model_name,
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "auc": metrics.get("auc"),
                    }
                )
    pd.DataFrame(overall_rows).to_csv(eval_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(by_bin_rows).to_csv(eval_dir / "by_bin_metrics.csv", index=False)
    pd.DataFrame(roc_data).T.to_csv(eval_dir / "roc_summary.csv")
    print(f"Benchmark outputs saved in: {eval_dir}")
    print(f"Plot outputs saved in: {plot_dir}")
    return payload
