from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

COLORS = {
    "MolDeBERTa": "#2E86AB",
    "MolDeBERTa x KG": "#06A77D",
    "MolFormer": "#A23B72",
    "NextGen": "#2E86AB",
    "SoftVoting": "#06A77D",
    "ECFP4": "#A23B72",
    "Morgan": "#F18F01",
}


def plot_auc_comparison_detailed(results: Dict, bins: List[str], out_dir: Path) -> None:
    models = ["SVM", "XGBoost", "MLP", "RandomForest"]
    methods = [k for k in ["MolDeBERTa", "MolDeBERTa x KG", "MolFormer"] if k in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, model in enumerate(models):
        ax = axes.flatten()[idx]
        x = np.arange(len(bins))
        width = 0.25
        for j, method in enumerate(methods):
            ys = [results[method]["by_bin"].get(b, {}).get(model, {}).get("auc", np.nan) for b in bins]
            ax.bar(x + (j - 1) * width, ys, width=width, label=method, color=COLORS.get(method, "#999999"), alpha=0.85)
        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "auc_comparison_detailed.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_active_prediction_counts(active_counts: Dict, bins: List[str], out_dir: Path) -> None:
    methods = list(active_counts.keys())
    x = np.arange(len(bins))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, method in enumerate(methods):
        y = [active_counts[method].get(b, 0) for b in bins]
        ax.bar(x + (j - 1) * width, y, width=width, label=method, color=COLORS.get(method, "#999999"), alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("Predicted Active Count")
    ax.set_title("Active Molecule Prediction Analysis")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "active_prediction_counts.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_detailed(results: Dict, bins: List[str], out_dir: Path) -> None:
    methods = [k for k in ["MolDeBERTa", "MolDeBERTa x KG", "MolFormer"] if k in results]
    mat = np.zeros((len(methods), len(bins)), dtype=np.float32)
    for i, method in enumerate(methods):
        for j, b in enumerate(bins):
            vals = [v["auc"] for v in results[method]["by_bin"].get(b, {}).values() if "auc" in v]
            mat[i, j] = float(np.mean(vals)) if vals else np.nan
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=0.9, xticklabels=bins, yticklabels=methods, ax=ax)
    ax.set_title("Performance Heatmap (with Soft Voting)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_detailed.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_detailed(results: Dict, bins: List[str], out_dir: Path) -> None:
    base = []
    improved = []
    for b in bins:
        b1 = [v["auc"] for v in results["MolDeBERTa"]["by_bin"].get(b, {}).values() if "auc" in v]
        b2 = [v["auc"] for v in results["MolDeBERTa x KG"]["by_bin"].get(b, {}).values() if "auc" in v]
        base.append(float(np.mean(b1)) if b1 else 0.0)
        improved.append(float(np.mean(b2)) if b2 else 0.0)
    imp = [(y - x) * 100 for x, y in zip(base, improved)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(bins)), imp, color=["#2ca02c" if v >= 0 else "#d62728" for v in imp])
    ax.axhline(0, color="black")
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_ylabel("AUC Improvement (points)")
    ax.set_title("Performance Improvements")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "improvement_detailed.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics_compairison(results: Dict, out_dir: Path) -> None:
    methods = [k for k in ["MolDeBERTa", "MolDeBERTa x KG", "MolFormer"] if k in results]
    metrics = ["accuracy", "f1", "auc"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, metric in enumerate(metrics):
        vals = [results[m]["overall"][metric] for m in methods]
        bar_colors = [COLORS.get(m, "#999999") for m in methods]
        axes[i].bar(methods, vals, color=bar_colors, alpha=0.88)
        axes[i].set_title(metric.upper())
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "model_metrics_compairison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(roc_data: Dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, d in roc_data.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{method} (AUC={d['auc']:.3f})", color=COLORS.get(method, "#444444"), linewidth=2.2)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend()
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
