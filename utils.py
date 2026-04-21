from __future__ import annotations

from typing import Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit


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


def stratified_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy().reset_index(drop=True)
    data["_scaffold"] = data[smiles_col].astype(str).map(_safe_scaffold)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, y=data[label_col], groups=data["_scaffold"]))
    train_df = data.iloc[train_idx].drop(columns=["_scaffold"]).reset_index(drop=True)
    test_df = data.iloc[test_idx].drop(columns=["_scaffold"]).reset_index(drop=True)
    return train_df, test_df
