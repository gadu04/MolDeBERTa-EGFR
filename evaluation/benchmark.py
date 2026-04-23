from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from tqdm.auto import tqdm

from config import CONFIG
from evaluation.fingerprints import compute_fps
from utils import detect_smiles_column, stratified_scaffold_split_3way


def _max_tanimoto_bins(train_smiles: list[str], valid_smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_fps = compute_fps(train_smiles)
    valid_fps = compute_fps(valid_smiles)
    sims = np.zeros((len(valid_fps),), dtype=np.float32)
    for i, fp in enumerate(tqdm(valid_fps, desc="Computing max Tanimoto", unit="mol")):
        if fp is None:
            sims[i] = 0.0
            continue
        vals = [DataStructs.TanimotoSimilarity(fp, tfp) for tfp in train_fps if tfp is not None]
        sims[i] = float(max(vals)) if vals else 0.0

    bins_cfg = CONFIG["TANIMOTO_BINS"]
    bidx = np.full((len(sims),), -1, dtype=np.int64)
    for idx, (lo, hi, _) in enumerate(bins_cfg):
        if idx == len(bins_cfg) - 1:
            mask = (sims >= lo) & (sims <= hi)
        else:
            mask = (sims >= lo) & (sims < hi)
        bidx[mask] = idx
    return sims, bidx


def _best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.01):
        pred = (prob >= t).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _metrics_at_threshold(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= thr).astype(int)
    return {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def run_benchmark() -> Dict:
    pred_dir = Path(CONFIG["PRED_DIR"])
    mol_valid_file = pred_dir / "mol_predictions_valid.csv"
    mol_test_file = pred_dir / "mol_predictions_test.csv"
    kg_valid_file = pred_dir / "kg_predictions_valid.csv"
    kg_test_file = pred_dir / "kg_predictions_test.csv"
    if not mol_valid_file.exists() or not mol_test_file.exists() or not kg_valid_file.exists() or not kg_test_file.exists():
        raise FileNotFoundError("Missing prediction files. Run finetuning and kg_training first.")

    mol_valid_df = pd.read_csv(mol_valid_file)
    mol_test_df = pd.read_csv(mol_test_file)
    kg_valid_df = pd.read_csv(kg_valid_file)
    kg_test_df = pd.read_csv(kg_test_file)

    merged_valid = mol_valid_df.merge(kg_valid_df[["smiles", "kg_score"]], on="smiles", how="inner")
    merged_test = mol_test_df.merge(kg_test_df[["smiles", "kg_score"]], on="smiles", how="inner")

    molformer_valid_file = pred_dir / "molformer_predictions_valid.csv"
    molformer_test_file = pred_dir / "molformer_predictions_test.csv"
    has_molformer = molformer_valid_file.exists() and molformer_test_file.exists()
    if has_molformer:
        mf_valid_df = pd.read_csv(molformer_valid_file)
        mf_test_df = pd.read_csv(molformer_test_file)
        if "molformer_score" in mf_valid_df.columns and "molformer_score" in mf_test_df.columns:
            merged_valid = merged_valid.merge(mf_valid_df[["smiles", "molformer_score"]], on="smiles", how="left")
            merged_test = merged_test.merge(mf_test_df[["smiles", "molformer_score"]], on="smiles", how="left")
        else:
            has_molformer = False
    if not has_molformer:
        print("MolFormer predictions not found -> benchmarking only MolDeBERTa and MolDeBERTa + KG.")
        print(f"Expected files: {molformer_valid_file}, {molformer_test_file}")
    if merged_valid.empty or merged_test.empty:
        raise RuntimeError("No overlapping smiles between mol and kg prediction files for valid/test.")

    data_df = pd.read_csv(CONFIG["DATA_CSV"])
    smiles_col = detect_smiles_column(data_df)
    label_col = CONFIG["LABEL_COLUMN"]
    train_df, valid_df, test_df = stratified_scaffold_split_3way(
        data_df,
        smiles_col,
        label_col,
        CONFIG["VALID_SIZE"],
        CONFIG["TEST_SIZE"],
        CONFIG["RANDOM_STATE"],
    )

    valid_ref = valid_df[[smiles_col]].rename(columns={smiles_col: "smiles"})
    test_ref = test_df[[smiles_col]].rename(columns={smiles_col: "smiles"})
    merged_valid = valid_ref.merge(merged_valid, on="smiles", how="left").dropna()
    merged_test = test_ref.merge(merged_test, on="smiles", how="left").dropna()

    y_valid = merged_valid["label"].to_numpy(dtype=np.int64)
    y_test = merged_test["label"].to_numpy(dtype=np.int64)
    mol_score_valid = merged_valid["mol_score"].to_numpy(dtype=np.float32)
    kg_score_valid = merged_valid["kg_score"].to_numpy(dtype=np.float32)
    mol_score_test = merged_test["mol_score"].to_numpy(dtype=np.float32)
    kg_score_test = merged_test["kg_score"].to_numpy(dtype=np.float32)

    # Phase 1 (valid): fit meta-learner + tune threshold on validation only.
    X_meta_valid = np.column_stack([mol_score_valid, kg_score_valid])
    meta = LogisticRegression(random_state=CONFIG["RANDOM_STATE"])
    meta.fit(X_meta_valid, y_valid)
    valid_meta_score = meta.predict_proba(X_meta_valid)[:, 1]
    thr = _best_threshold(y_valid, valid_meta_score)
    mol_thr = _best_threshold(y_valid, mol_score_valid)

    # Phase 2 (test): report metrics/charts strictly on test only.
    X_meta_test = np.column_stack([mol_score_test, kg_score_test])
    final_score_test = meta.predict_proba(X_meta_test)[:, 1]
    method_scores: Dict[str, np.ndarray] = {
        "MolDeBERTa": mol_score_test,
        "MolDeBERTa + KG": final_score_test,
    }
    if has_molformer and "molformer_score" in merged_test.columns:
        method_scores["MolFormer"] = merged_test["molformer_score"].to_numpy(dtype=np.float32)

    _, bin_idx = _max_tanimoto_bins(
        train_df[smiles_col].astype(str).tolist(),
        merged_test["smiles"].astype(str).tolist(),
    )
    bin_names = [b[2] for b in CONFIG["TANIMOTO_BINS"]]

    rows = []
    by_bin_auc: Dict[str, list[float]] = {name: [] for name in method_scores}
    for idx, bname in enumerate(bin_names):
        mask = bin_idx == idx
        row = {"bin": bname, "n_samples": int(mask.sum())}
        for method_name, score in method_scores.items():
            if mask.sum() < 3 or len(np.unique(y_test[mask])) < 2:
                auc_val = float("nan")
            else:
                auc_val = float(roc_auc_score(y_test[mask], score[mask]))
            by_bin_auc[method_name].append(auc_val)
            row[f"auc_{method_name.lower().replace(' ', '_').replace('+', 'plus')}"] = auc_val
        rows.append(row)

    summary: Dict[str, Dict[str, float] | list] = {
        "MolDeBERTa": _metrics_at_threshold(y_test, mol_score_test, mol_thr),
        "Pipeline_MolDeBERTa_plus_KG": _metrics_at_threshold(y_test, final_score_test, thr),
        "threshold_source": "validation_set",
        "benchmark_set": "test_set",
        "meta_learner_coef": meta.coef_.tolist(),
        "meta_learner_intercept": meta.intercept_.tolist(),
    }
    if "MolFormer" in method_scores:
        mf_thr = _best_threshold(y_valid, merged_valid["molformer_score"].to_numpy(dtype=np.float32))
        summary["MolFormer"] = _metrics_at_threshold(y_test, method_scores["MolFormer"], mf_thr)

    kg_coef = float(meta.coef_[0][1]) if meta.coef_.ndim == 2 else float(meta.coef_[1])
    if abs(kg_coef) < 0.05:
        print(
            f"WARNING: Meta-learner gives near-zero weight to KG (coef={kg_coef:.6f}). "
            "This usually means KG scores add little information over MolDeBERTa on current split."
        )

    eval_dir = Path(CONFIG["EVAL_DIR"])
    plot_dir = Path(CONFIG["PLOT_DIR"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(eval_dir / "scaffold_hopping_auc_by_bin.csv", index=False)
    summary_rows = [
        {"model": "MolDeBERTa", **summary["MolDeBERTa"]},
        {"model": "Pipeline_MolDeBERTa_plus_KG", **summary["Pipeline_MolDeBERTa_plus_KG"]},
    ]
    pd.DataFrame(summary_rows).to_csv(eval_dir / "benchmark_summary.csv", index=False)
    with open(eval_dir / "meta_learner_weights.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "coef": summary["meta_learner_coef"],
                "intercept": summary["meta_learner_intercept"],
            },
            f,
            indent=2,
        )
    with open(eval_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "by_bin": rows}, f, indent=2)

    # Additional plots expected in output/plots
    _plot_extra_benchmark_figures(
        plot_dir=plot_dir,
        bin_names=bin_names,
        bin_idx=bin_idx,
        y_true=y_test,
        method_scores=method_scores,
        mol_threshold=mol_thr,
        final_threshold=thr,
        by_bin_auc=by_bin_auc,
        summary=summary,
    )
    generated_plots = [
        "auc_comparison_detailed.png",
        "heatmap_detailed.png",
        "improvement_detailed.png",
        "active_prediction_counts.png",
        "model_metrics_compairison.png",
        "roc_comparison.png",
    ]

    x = np.arange(len(bin_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_methods = list(method_scores.keys())
    offsets = np.linspace(-width, width, len(bar_methods))
    colors = {"MolDeBERTa": "#2E86AB", "MolDeBERTa + KG": "#06A77D", "MolFormer": "#A23B72"}
    for m, off in zip(bar_methods, offsets):
        ax.bar(x + off / 2, by_bin_auc[m], width=max(0.2, 0.6 / len(bar_methods)), label=m, color=colors.get(m, "#999999"))
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Tanimoto Similarity Bin")
    ax.set_title("So sanh ROC-AUC theo Similarity Bin")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "scaffold_hopping_auc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved benchmark summary: {eval_dir / 'benchmark_summary.csv'}")
    print(f"Saved bin AUC report: {eval_dir / 'scaffold_hopping_auc_by_bin.csv'}")
    print(f"Saved chart: {plot_dir / 'scaffold_hopping_auc_comparison.png'}")
    for name in generated_plots:
        print(f"Saved chart: {plot_dir / name}")
    return {"summary": summary, "by_bin": rows}


def _plot_extra_benchmark_figures(
    plot_dir: Path,
    bin_names: list[str],
    bin_idx: np.ndarray,
    y_true: np.ndarray,
    method_scores: Dict[str, np.ndarray],
    mol_threshold: float,
    final_threshold: float,
    by_bin_auc: Dict[str, list[float]],
    summary: Dict,
) -> None:
    x = np.arange(len(bin_names))
    width = 0.36
    colors = {"MolDeBERTa": "#2E86AB", "MolDeBERTa + KG": "#06A77D", "MolFormer": "#A23B72"}

    # auc_comparison_detailed.png
    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = list(by_bin_auc.keys())
    offsets = np.linspace(-width, width, len(methods))
    for m, off in zip(methods, offsets):
        ax.bar(x + off / 2, by_bin_auc[m], width=max(0.2, 0.6 / len(methods)), label=m, color=colors.get(m, "#999999"))
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("Detailed AUC Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "auc_comparison_detailed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # heatmap_detailed.png
    heat = np.vstack([by_bin_auc[m] for m in methods])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)
    ax.set_xticks(np.arange(len(bin_names)))
    ax.set_xticklabels(bin_names)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Performance Heatmap")
    for i in range(len(methods)):
        for j in range(len(bin_names)):
            val = heat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(plot_dir / "heatmap_detailed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # improvement_detailed.png
    ref_name = "MolDeBERTa"
    cmp_name = "MolDeBERTa + KG" if "MolDeBERTa + KG" in by_bin_auc else methods[-1]
    imp = [(p - m) * 100 for m, p in zip(by_bin_auc[ref_name], by_bin_auc[cmp_name])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, imp, color=["#2ca02c" if v >= 0 else "#d62728" for v in imp], alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Performance Improvements")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "improvement_detailed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # active_prediction_counts.png
    thresholds = {"MolDeBERTa": mol_threshold, "MolDeBERTa + KG": final_threshold}
    active_by_method: Dict[str, list[int]] = {}
    for name, score in method_scores.items():
        thr = thresholds.get(name, _best_threshold(y_true, score))
        pred = (score >= thr).astype(int)
        vals = []
        for i in range(len(bin_names)):
            mask = bin_idx == i
            vals.append(int(pred[mask].sum()))
        active_by_method[name] = vals
    fig, ax = plt.subplots(figsize=(10, 5))
    methods_active = list(active_by_method.keys())
    offsets = np.linspace(-width, width, len(methods_active))
    for m, off in zip(methods_active, offsets):
        ax.bar(x + off / 2, active_by_method[m], width=max(0.2, 0.6 / len(methods_active)), color=colors.get(m, "#999999"), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names)
    ax.set_ylabel("Predicted Active Count")
    ax.set_title("Active Prediction Counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "active_prediction_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # model_metrics_compairison.png
    metrics = ["accuracy", "f1_macro", "mcc", "roc_auc"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    summary_method_map = {
        "MolDeBERTa": "MolDeBERTa",
        "MolDeBERTa + KG": "Pipeline_MolDeBERTa_plus_KG",
        "MolFormer": "MolFormer",
    }
    methods_metric = [m for m in method_scores.keys() if summary_method_map.get(m) in summary]
    for i, metric in enumerate(metrics):
        vals = [summary[summary_method_map[m]][metric] for m in methods_metric]
        cols = [colors.get(m, "#999999") for m in methods_metric]
        axes[i].bar(methods_metric, vals, color=cols)
        axes[i].set_title(metric)
        axes[i].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "model_metrics_compairison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # roc_comparison.png
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, score in method_scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_val = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, label=f"{m} (AUC={auc_val:.4f})", color=colors.get(m, "#999999"))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "roc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
