"""KG statistical feature extraction for classification."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from neo4j import GraphDatabase
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from config import CONFIG

_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=2048)


def _run(session, query: str, params: Dict | None = None):
    return session.run(query, params or {})


def _get_available_labels(session) -> set[str]:
    rows = _run(session, "CALL db.labels() YIELD label RETURN label")
    return {str(r["label"]) for r in rows}


def _feature_one(session, smiles: str, target_name: str, labels: set[str]) -> List[float]:
    """
    Extract richer count-based graph features (no transductive embeddings):
    [num_warheads, num_moas, has_egfr_path, deg_total, num_scaffolds,
     num_targets, num_interaction_groups, num_functional_groups]
    Falls back to 0 when molecule not found/isolated.
    """
    try:
        if "Molecule" not in labels:
            return [0.0] * 8

        if "Warhead" in labels:
            row_w = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(w:Warhead)
                RETURN coalesce(count(DISTINCT w), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_warheads = float(int(row_w["c"])) if row_w is not None else 0.0
        elif "FunctionalGroup" in labels:
            row_w = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(wf:FunctionalGroup)
                RETURN coalesce(count(DISTINCT wf), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_warheads = float(int(row_w["c"])) if row_w is not None else 0.0
        else:
            num_warheads = 0.0

        if "MoA" in labels and "Interaction_Group" in labels:
            row_moa = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(:Interaction_Group)-[]-(moa:MoA)
                RETURN coalesce(count(DISTINCT moa), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_moas = float(int(row_moa["c"])) if row_moa is not None else 0.0
        elif "MoA" in labels:
            row_moa = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(moa:MoA)
                RETURN coalesce(count(DISTINCT moa), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_moas = float(int(row_moa["c"])) if row_moa is not None else 0.0
        else:
            num_moas = 0.0

        if "Target" in labels:
            row_egfr = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH p=(m)-[*1..3]-(:Target {name: $target_name})
                RETURN CASE WHEN p IS NULL THEN 0 ELSE 1 END AS has_path
                """,
                {"smiles": smiles, "target_name": target_name},
            ).single()
            has_egfr_path = float(int(row_egfr["has_path"])) if row_egfr is not None else 0.0
        else:
            has_egfr_path = 0.0

        row_deg = _run(
            session,
            """
            MATCH (m:Molecule {smiles: $smiles})
            RETURN coalesce(size((m)--()), 0) AS deg_total
            """,
            {"smiles": smiles},
        ).single()
        deg_total = float(int(row_deg["deg_total"])) if row_deg is not None else 0.0

        if "Scaffold" in labels:
            row_scaf = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(s:Scaffold)
                RETURN coalesce(count(DISTINCT s), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_scaffolds = float(int(row_scaf["c"])) if row_scaf is not None else 0.0
        else:
            num_scaffolds = 0.0

        if "Target" in labels:
            row_t = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(t:Target)
                RETURN coalesce(count(DISTINCT t), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_targets = float(int(row_t["c"])) if row_t is not None else 0.0
        else:
            num_targets = 0.0

        if "Interaction_Group" in labels:
            row_ig = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(ig:Interaction_Group)
                RETURN coalesce(count(DISTINCT ig), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_interaction_groups = float(int(row_ig["c"])) if row_ig is not None else 0.0
        else:
            num_interaction_groups = 0.0

        if "FunctionalGroup" in labels:
            row_fg = _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[]-(fg:FunctionalGroup)
                RETURN coalesce(count(DISTINCT fg), 0) AS c
                """,
                {"smiles": smiles},
            ).single()
            num_functional_groups = float(int(row_fg["c"])) if row_fg is not None else 0.0
        else:
            num_functional_groups = 0.0

        return [
            num_warheads,
            num_moas,
            has_egfr_path,
            deg_total,
            num_scaffolds,
            num_targets,
            num_interaction_groups,
            num_functional_groups,
        ]
    except Exception:
        return [0.0] * 8


def build_kg_statistical_features(smiles_list: List[str], target_name: str = "EGFR") -> np.ndarray:
    uri = CONFIG["NEO4J_URI"]
    user = CONFIG["NEO4J_USER"]
    pwd = CONFIG["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    out = np.zeros((len(smiles_list), 8), dtype=np.float32)
    try:
        with driver.session() as session:
            labels = _get_available_labels(session)
            for i, smiles in enumerate(smiles_list):
                out[i] = np.array(_feature_one(session, str(smiles), target_name, labels), dtype=np.float32)
    finally:
        driver.close()
    return out


def _fingerprint_or_none(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def infer_valid_features_by_knn(
    train_smiles: List[str],
    train_features: np.ndarray,
    valid_smiles: List[str],
    top_k: int = 5,
) -> np.ndarray:
    """
    Infer valid KG features from Top-K structurally similar train molecules.
    Uses BulkTanimotoSimilarity for speed.
    """
    if len(train_smiles) == 0:
        return np.zeros((len(valid_smiles), train_features.shape[1] if train_features.ndim == 2 else 8), dtype=np.float32)

    train_fps = [_fingerprint_or_none(s) for s in train_smiles]
    valid_fps = [_fingerprint_or_none(s) for s in valid_smiles]

    dim = train_features.shape[1]
    out = np.zeros((len(valid_smiles), dim), dtype=np.float32)
    nonzero_mask = np.any(train_features != 0, axis=1)
    global_mean = train_features[nonzero_mask].mean(axis=0) if nonzero_mask.any() else np.zeros((dim,), dtype=np.float32)

    for i, vfp in enumerate(valid_fps):
        if vfp is None:
            out[i] = global_mean
            continue

        sim_vals = np.array(
            DataStructs.BulkTanimotoSimilarity(vfp, [fp for fp in train_fps if fp is not None]),
            dtype=np.float32,
        )
        valid_idx = np.array([idx for idx, fp in enumerate(train_fps) if fp is not None], dtype=np.int64)
        if sim_vals.size == 0:
            out[i] = global_mean
            continue

        k = int(max(1, min(top_k, sim_vals.size)))
        top_pos = np.argpartition(sim_vals, -k)[-k:]
        top_train_idx = valid_idx[top_pos]
        top_sims = sim_vals[top_pos]
        w = top_sims / max(float(top_sims.sum()), 1e-8)
        out[i] = (train_features[top_train_idx] * w[:, None]).sum(axis=0).astype(np.float32)
    return out


def _knn_similarity_stats(
    train_fps: List,
    query_fps: List,
    top_k: int = 5,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Return per-query [mean_sim, max_sim, std_sim] from Top-K similarities.
    """
    stats = np.zeros((len(query_fps), 3), dtype=np.float32)
    valid_train = [fp for fp in train_fps if fp is not None]
    valid_train_idx = np.array([i for i, fp in enumerate(train_fps) if fp is not None], dtype=np.int64)
    if len(valid_train) == 0:
        return stats

    for i, qfp in enumerate(query_fps):
        if qfp is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, valid_train), dtype=np.float32)
        if exclude_self and i < len(query_fps):
            same_pos = np.where(valid_train_idx == i)[0]
            if same_pos.size > 0:
                sims[same_pos[0]] = -1.0
        k = int(max(1, min(top_k, np.sum(sims >= 0))))
        if k <= 0:
            continue
        top = np.partition(sims, -k)[-k:]
        top = top[top >= 0]
        if top.size == 0:
            continue
        stats[i, 0] = float(np.mean(top))
        stats[i, 1] = float(np.max(top))
        stats[i, 2] = float(np.std(top))
    return stats


def build_train_and_valid_kg_features(
    train_smiles: List[str],
    valid_smiles: List[str],
    target_name: str = "EGFR",
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = build_kg_statistical_features(train_smiles, target_name=target_name)
    train_fps = [_fingerprint_or_none(s) for s in train_smiles]
    valid_fps = [_fingerprint_or_none(s) for s in valid_smiles]

    valid_features = infer_valid_features_by_knn(
        train_smiles=train_smiles,
        train_features=train_features,
        valid_smiles=valid_smiles,
        top_k=top_k,
    )
    # Add KNN similarity statistics to enrich both train/valid representations.
    train_sim_stats = _knn_similarity_stats(train_fps=train_fps, query_fps=train_fps, top_k=top_k, exclude_self=True)
    valid_sim_stats = _knn_similarity_stats(train_fps=train_fps, query_fps=valid_fps, top_k=top_k, exclude_self=False)

    train_out = np.concatenate([train_features, train_sim_stats], axis=1)
    valid_out = np.concatenate([valid_features, valid_sim_stats], axis=1)
    return train_out, valid_out
