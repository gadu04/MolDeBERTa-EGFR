from __future__ import annotations

from typing import Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
import numpy as np


def detect_smiles_column(df: pd.DataFrame) -> str:
    cols_upper = {c.upper(): c for c in df.columns}
    for candidate in ["SMILES", "SELFIES", "CANONICAL_SMILES", "SMILE"]:
        if candidate in cols_upper:
            return cols_upper[candidate]
    return df.columns[0]


def _safe_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid::{smiles}"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return Chem.MolToSmiles(mol)


def stratified_scaffold_split_3way(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    valid_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy().reset_index(drop=True)
    data["_scaffold"] = data[smiles_col].astype(str).map(_safe_scaffold)
    total_holdout = float(valid_size) + float(test_size)
    if total_holdout <= 0.0 or total_holdout >= 1.0:
        raise ValueError("valid_size + test_size must be in (0, 1).")

    splitter_outer = GroupShuffleSplit(n_splits=1, test_size=total_holdout, random_state=seed)
    train_idx, holdout_idx = next(splitter_outer.split(data, y=data[label_col], groups=data["_scaffold"]))
    train_df = data.iloc[train_idx].copy().reset_index(drop=True)
    holdout_df = data.iloc[holdout_idx].copy().reset_index(drop=True)

    # Split holdout into valid/test by scaffolds (preserves zero-leakage across groups).
    valid_frac_in_holdout = float(valid_size) / float(total_holdout)
    splitter_inner = GroupShuffleSplit(n_splits=1, test_size=(1.0 - valid_frac_in_holdout), random_state=seed + 1)
    valid_idx_local, test_idx_local = next(
        splitter_inner.split(holdout_df, y=holdout_df[label_col], groups=holdout_df["_scaffold"])
    )
    valid_df = holdout_df.iloc[valid_idx_local].copy().reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx_local].copy().reset_index(drop=True)

    train_df = train_df.drop(columns=["_scaffold"])
    valid_df = valid_df.drop(columns=["_scaffold"])
    test_df = test_df.drop(columns=["_scaffold"])
    return train_df, valid_df, test_df


def stratified_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Backward-compatible wrapper for legacy callers (train/test only).
    data = df.copy().reset_index(drop=True)
    data["_scaffold"] = data[smiles_col].astype(str).map(_safe_scaffold)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, y=data[label_col], groups=data["_scaffold"]))
    train_df = data.iloc[train_idx].drop(columns=["_scaffold"]).reset_index(drop=True)
    test_df = data.iloc[test_idx].drop(columns=["_scaffold"]).reset_index(drop=True)
    return train_df, test_df


def labels_to_int(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.lower().str.strip()
    return values.map(lambda x: 1 if x in {"1", "active", "true", "yes"} else 0).to_numpy(dtype=np.int64)
