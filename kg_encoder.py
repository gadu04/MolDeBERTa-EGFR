"""Knowledge Graph Encoder for molecular property prediction."""

import sys
from pathlib import Path

# Ensure config can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from typing import Dict, List
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
            # ✅ Step 1: Check if graph projection exists
            print("      Checking graph projection...")
            result = _run(session, "CALL gds.graph.exists('egfr_kg') YIELD exists RETURN exists")
            exists = result.single()["exists"]
            
            if exists:
                print("      Dropping existing projection...")
                _run(session, "CALL gds.graph.drop('egfr_kg')")
            
            # ✅ Step 2: Create new projection
            print("      Creating graph projection...")
            _run(
                session,
                """
                CALL gds.graph.project(
                    'egfr_kg',
                    ['Molecule','Scaffold','FunctionalGroup','Warhead','Target','MoA'],
                    {
                      HAS_SCAFFOLD: {type:'HAS_SCAFFOLD', orientation:'UNDIRECTED'},
                      HAS_FUNCTIONAL_GROUP: {type:'HAS_FUNCTIONAL_GROUP', orientation:'UNDIRECTED'},
                      CONTAINS_WARHEAD: {type:'CONTAINS_WARHEAD', orientation:'UNDIRECTED'},
                      TESTED_AGAINST: {type:'TESTED_AGAINST', orientation:'UNDIRECTED'},
                      ACTS_VIA: {type:'ACTS_VIA', orientation:'UNDIRECTED'}
                    }
                )
                """
            )
            print("      ✅ Graph projection created")

            # ✅ Step 3: Compute and mutate FastRP embeddings
            print(f"      Computing FastRP embeddings (dim={dim})...")
            _run(
                session,
                """
                CALL gds.fastRP.mutate(
                  'egfr_kg',
                  {
                    embeddingDimension: $dim, 
                    mutateProperty: 'kg_emb',
                    randomSeed: 42
                  }
                )
                """,
                {"dim": dim},
            )
            print("      ✅ FastRP embeddings computed")
            
            # ✅ Step 4: Write embeddings back to nodes
            print("      Writing embeddings to nodes...")
            _run(
                session,
                """
                CALL gds.graph.nodeProperties.write(
                  'egfr_kg',
                  ['kg_emb']
                )
                """
            )
            print("      ✅ Embeddings written to nodes")
            
            # ✅ Step 5: Verify embeddings exist
            result = _run(
                session,
                """
                MATCH (m:Molecule)
                WHERE m.kg_emb IS NOT NULL
                RETURN count(m) AS count
                """
            )
            count = result.single()["count"]
            print(f"      ✅ Verified: {count} molecules have kg_emb property")
    
    except Exception as e:
        print(f"      ❌ Error creating KG embeddings: {e}")
        raise
    
    finally:
        driver.close()


def fetch_kg_embeddings(smiles_list: List[str]) -> Dict[str, List[float]]:
    """Fetch precomputed embeddings for Molecule nodes."""
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
            for r in rows:
                if r["emb"] is not None:
                    result[r["smiles"]] = r["emb"]
    
    finally:
        driver.close()
    
    return result


def build_kg_embeddings_for_df(df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    """Return KG embeddings aligned to df order."""
    dim = CONFIG["KG_EMBED_DIM"]
    smiles_list = df[smiles_col].astype(str).tolist()
    emb_map = fetch_kg_embeddings(smiles_list)

    # ✅ Count missing embeddings
    missing_count = len(smiles_list) - len(emb_map)
    if missing_count > 0:
        print(f"      ⚠️  {missing_count} molecules don't have KG embeddings (will use zeros)")

    out = np.zeros((len(smiles_list), dim), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        if s in emb_map:
            out[i] = np.array(emb_map[s], dtype=np.float32)
    
    return out


def augment_embeddings_with_kg(
    nextgen_embeddings: np.ndarray,
    kg_embeddings: np.ndarray,
    fusion_type: str = "concat"
) -> np.ndarray:
    """
    Augment NextGen embeddings with KG embeddings.
    
    Args:
        nextgen_embeddings: [N, 768]
        kg_embeddings: [N, 128]
        fusion_type: "concat" | "add" | "weighted"
    
    Returns:
        Augmented embeddings: [N, D_augmented]
    """
    if fusion_type == "concat":
        # [N, 768] + [N, 128] → [N, 896]
        return np.concatenate([nextgen_embeddings, kg_embeddings], axis=1)
    
    elif fusion_type == "add":
        # Resize KG to match NextGen dim, then add
        if kg_embeddings.shape[1] != nextgen_embeddings.shape[1]:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            kg_scaled = scaler.fit_transform(kg_embeddings)
            # Simple linear projection
            w = np.random.randn(kg_embeddings.shape[1], nextgen_embeddings.shape[1]) * 0.01
            kg_proj = kg_scaled @ w
        else:
            kg_proj = kg_embeddings
        return nextgen_embeddings + kg_proj
    
    elif fusion_type == "weighted":
        # 0.8 * NextGen + 0.2 * KG
        kg_scaled = (kg_embeddings - kg_embeddings.mean(axis=0)) / (kg_embeddings.std(axis=0) + 1e-8)
        kg_scaled = kg_scaled * nextgen_embeddings.std(axis=0)
        kg_scaled = kg_scaled + nextgen_embeddings.mean(axis=0)
        return 0.8 * nextgen_embeddings + 0.2 * kg_scaled
    
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")