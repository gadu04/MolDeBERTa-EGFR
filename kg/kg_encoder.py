"""Knowledge Graph Encoder for molecular property prediction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from config import CONFIG


def _run(session, query: str, params: Dict = None):
    return session.run(query, params or {})


def ensure_kg_embeddings():
    """Compute FastRP embeddings in Neo4j (GDS)."""
    uri = CONFIG["NEO4J_URI"]
    user = CONFIG["NEO4J_USER"]
    pwd = CONFIG["NEO4J_PASSWORD"]
    dim = CONFIG["KG_EMBED_DIM"]
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session() as session:
            result = _run(session, "CALL gds.graph.exists('egfr_kg') YIELD exists RETURN exists")
            exists = result.single()["exists"]
            if exists:
                _run(
                    session,
                    """
                    CALL gds.graph.drop('egfr_kg')
                    YIELD graphName
                    RETURN graphName
                    """,
                )
            _run(
                session,
                """
                CALL gds.graph.project(
                    'egfr_kg',
                    ['Molecule','Scaffold','FunctionalGroup','Interaction_Group','Target','MoA'],
                    {
                      HAS_SCAFFOLD: {type:'HAS_SCAFFOLD', orientation:'UNDIRECTED'},
                      HAS_FUNCTIONAL_GROUP: {type:'HAS_FUNCTIONAL_GROUP', orientation:'UNDIRECTED'},
                      HAS_INTERACTION_GROUP: {type:'HAS_INTERACTION_GROUP', orientation:'UNDIRECTED'},
                      TESTED_AGAINST: {type:'TESTED_AGAINST', orientation:'UNDIRECTED'},
                      ACTS_VIA: {type:'ACTS_VIA', orientation:'UNDIRECTED'}
                    }
                )
                """,
            )
            _run(
                session,
                """
                CALL gds.fastRP.mutate('egfr_kg', {
                    embeddingDimension: $dim,
                    mutateProperty: 'kg_emb',
                    randomSeed: 42
                })
                """,
                {"dim": dim},
            )
            _run(
                session,
                """
                CALL gds.graph.nodeProperties.write('egfr_kg', ['kg_emb'])
                """,
            )
    finally:
        driver.close()


def fetch_kg_embeddings(smiles_list: List[str]) -> Dict[str, List[float]]:
    uri = CONFIG["NEO4J_URI"]
    user = CONFIG["NEO4J_USER"]
    pwd = CONFIG["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    result: Dict[str, List[float]] = {}
    try:
        with driver.session() as session:
            rows = _run(
                session,
                """
                MATCH (m:Molecule)
                WHERE m.smiles IN $smiles AND m.kg_emb IS NOT NULL
                RETURN m.smiles AS smiles, m.kg_emb AS emb
                """,
                {"smiles": smiles_list},
            )
            for row in rows:
                if row["emb"] is not None:
                    result[row["smiles"]] = row["emb"]
    finally:
        driver.close()
    return result


def infer_test_embedding_with_injection(smiles: str) -> np.ndarray:
    uri = CONFIG["NEO4J_URI"]
    user = CONFIG["NEO4J_USER"]
    pwd = CONFIG["NEO4J_PASSWORD"]
    dim = CONFIG["KG_EMBED_DIM"]
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session() as session:
            _run(
                session,
                """
                MERGE (m:Molecule {smiles: $smiles})
                SET m.source = 'TestInference', m.is_virtual = false
                WITH m
                MATCH (s:Scaffold)<-[:HAS_SCAFFOLD]-(:Molecule)
                WITH m, s LIMIT 1
                MERGE (m)-[:HAS_SCAFFOLD]->(s)
                WITH m
                MATCH (t:Target) WITH m, t LIMIT 1
                MERGE (m)-[:TESTED_AGAINST]->(t)
                """,
                {"smiles": smiles},
            )
            rows = _run(
                session,
                """
                CALL gds.fastRP.stream('egfr_kg', {embeddingDimension: $dim, randomSeed: 42})
                YIELD nodeId, embedding
                WITH gds.util.asNode(nodeId) AS n, embedding
                WHERE n:Molecule AND n.smiles = $smiles
                RETURN embedding
                """,
                {"smiles": smiles, "dim": dim},
            )
            rec = rows.single()
            emb = np.array(rec["embedding"], dtype=np.float32) if rec and rec["embedding"] is not None else np.zeros((dim,), dtype=np.float32)
            _run(
                session,
                """
                MATCH (m:Molecule {smiles: $smiles})
                WHERE m.source = 'TestInference'
                DETACH DELETE m
                """,
                {"smiles": smiles},
            )
            return emb
    finally:
        driver.close()


def build_kg_embeddings_for_df(df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    dim = CONFIG["KG_EMBED_DIM"]
    smiles_list = df[smiles_col].astype(str).tolist()
    emb_map = fetch_kg_embeddings(smiles_list)
    out = np.zeros((len(smiles_list), dim), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        if smiles in emb_map:
            out[i] = np.array(emb_map[smiles], dtype=np.float32)
    return out


def augment_embeddings_with_kg(
    nextgen_embeddings: np.ndarray,
    kg_embeddings: np.ndarray,
    fusion_type: str = "concat",
) -> np.ndarray:
    if fusion_type == "concat":
        return np.concatenate([nextgen_embeddings, kg_embeddings], axis=1)
    if fusion_type == "add":
        if kg_embeddings.shape[1] != nextgen_embeddings.shape[1]:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            kg_scaled = scaler.fit_transform(kg_embeddings)
            w = np.random.randn(kg_embeddings.shape[1], nextgen_embeddings.shape[1]) * 0.01
            kg_proj = kg_scaled @ w
        else:
            kg_proj = kg_embeddings
        return nextgen_embeddings + kg_proj
    if fusion_type == "weighted":
        kg_scaled = (kg_embeddings - kg_embeddings.mean(axis=0)) / (kg_embeddings.std(axis=0) + 1e-8)
        kg_scaled = kg_scaled * nextgen_embeddings.std(axis=0)
        kg_scaled = kg_scaled + nextgen_embeddings.mean(axis=0)
        return 0.8 * nextgen_embeddings + 0.2 * kg_scaled
    raise ValueError(f"Unknown fusion_type: {fusion_type}")
