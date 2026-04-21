from __future__ import annotations

from typing import Dict

import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings(
    "ignore",
    message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
    category=UserWarning,
)


def get_models() -> Dict[str, object]:
    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            tree_method="hist",
            device="cuda:0",
            predictor="gpu_predictor",
            max_bin=256,
            random_state=42,
            n_jobs=1,
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }


def fit_predict_proba(model, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = np.ascontiguousarray(scaler.fit_transform(X_train).astype(np.float32))
    Xva = np.ascontiguousarray(scaler.transform(X_valid).astype(np.float32))
    model.fit(Xtr, y_train)
    return model.predict_proba(Xva)[:, 1]


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
