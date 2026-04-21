from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=2048)


def compute_ecfp4(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if radius != 2 or n_bits != 2048:
        return GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def compute_fps(smiles_list: List[str]) -> List:
    return [compute_ecfp4(str(s)) for s in smiles_list]


def max_train_similarity(valid_fps: List, train_fps: List) -> np.ndarray:
    out = np.zeros((len(valid_fps),), dtype=np.float32)
    for i, v in enumerate(valid_fps):
        if v is None:
            out[i] = 0.0
            continue
        sims = [DataStructs.TanimotoSimilarity(v, t) for t in train_fps if t is not None]
        out[i] = float(max(sims)) if sims else 0.0
    return out


def assign_bins(similarities: np.ndarray, bins: List[Tuple[float, float, str]]) -> np.ndarray:
    assigned = np.full((len(similarities),), -1, dtype=np.int64)
    for idx, (lo, hi, _) in enumerate(bins):
        if idx == len(bins) - 1:
            mask = (similarities >= lo) & (similarities <= hi)
        else:
            mask = (similarities >= lo) & (similarities < hi)
        assigned[mask] = idx
    return assigned
