"""Data loader helpers for evaluation pipeline."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import CONFIG
from utils import detect_smiles_column, stratified_scaffold_split_3way


def load_data_splits() -> Dict:
    df = pd.read_csv(CONFIG["DATA_CSV"])
    smiles_col = detect_smiles_column(df)
    label_col = CONFIG["LABEL_COLUMN"]
    train_df, valid_df, test_df = stratified_scaffold_split_3way(
        df,
        smiles_col,
        label_col,
        CONFIG["VALID_SIZE"],
        CONFIG["TEST_SIZE"],
        CONFIG["RANDOM_STATE"],
    )
    y_train = train_df[label_col].astype(str).str.lower().map(lambda x: 1 if x in {"active", "1", "true", "yes"} else 0).to_numpy()
    y_valid = valid_df[label_col].astype(str).str.lower().map(lambda x: 1 if x in {"active", "1", "true", "yes"} else 0).to_numpy()
    y_test = test_df[label_col].astype(str).str.lower().map(lambda x: 1 if x in {"active", "1", "true", "yes"} else 0).to_numpy()
    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "smiles_col": smiles_col,
    }


def load_saved_embeddings() -> Dict:
    import pathlib
    root = pathlib.Path(CONFIG["OUTPUT_DIR"])
    return {
        "bert_train": np.load(root / "bert_train.npy"),
        "bert_valid": np.load(root / "bert_valid.npy"),
        "kg_train": np.load(root / "kg_train.npy"),
        "kg_valid": np.load(root / "kg_valid.npy"),
    }
