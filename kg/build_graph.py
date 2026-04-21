"""Standalone Knowledge Graph builder entrypoint.

This file contains the full build pipeline so `python scripts/build_kg.py`
can delegate to a single self-contained module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rdkit import Chem, DataStructs
from rdkit.Chem import Fragments
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import stratified_scaffold_split
from config import CONFIG, resolve_domain_config_path


load_dotenv()

PROJECT_ROOT = Path(CONFIG.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

NEO4J_URI = CONFIG["NEO4J_URI"]
NEO4J_USER = CONFIG["NEO4J_USER"]
NEO4J_PASSWORD = CONFIG["NEO4J_PASSWORD"]

EXPERIMENTAL_CSV = PROJECT_ROOT / "data" / "raw" / "data_end.csv"
if not EXPERIMENTAL_CSV.exists():
    EXPERIMENTAL_CSV = Path(CONFIG.get("KG_EXPERIMENTAL_CSV", EXPERIMENTAL_CSV))

DENOVO_CSV = PROJECT_ROOT / "data" / "raw" / "DeNovo_Molecule.csv"
BATCH_SIZE = int(CONFIG.get("KG_BUILD_BATCH_SIZE", 1000))

_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=1024)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        pass
    return None


def get_scaffold(mol: Chem.Mol) -> Optional[str]:
    if not mol:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core) if core else Chem.MolToSmiles(mol)
    except Exception:
        return Chem.MolToSmiles(mol)


def get_ecfp4(smiles_list: List[str], n_bits: int = 1024):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fingerprints.append(_MORGAN_GENERATOR.GetFingerprint(mol))
        else:
            fingerprints.append([0] * n_bits)
    return fingerprints


class ChemistryAnalyzer:
    def __init__(self, config: Dict) -> None:
        target_info = config["Target_Info"]
        self._generic_label: str = target_info["generic_label"]
        self._similarity_threshold: float = float(target_info["similarity_threshold"])

        self._ref_fps: List[Dict] = []
        for name, info in config["Reference_Drugs"].items():
            mol = Chem.MolFromSmiles(info["smiles"])
            if mol:
                self._ref_fps.append(
                    {
                        "name": name,
                        "fp": _MORGAN_GENERATOR.GetFingerprint(mol),
                        "target": info["target"],
                    }
                )

        self._interaction_groups: List[Dict] = []
        for entry in config["Interaction_Groups"]:
            pattern = Chem.MolFromSmarts(entry["smarts"])
            if pattern:
                self._interaction_groups.append(
                    {
                        "name": entry["name"],
                        "pattern": pattern,
                        "moa": entry["moa"],
                    }
                )

        self._target_rules: List[Tuple[str, Chem.Mol]] = []
        for name, smarts in config["Target_Specific_Rules"].items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                self._target_rules.append((name, pattern))

    def assign_target(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return self._generic_label

        query_fp = _MORGAN_GENERATOR.GetFingerprint(mol)
        best_target = self._generic_label
        max_sim = 0.0

        for ref in self._ref_fps:
            sim = DataStructs.TanimotoSimilarity(query_fp, ref["fp"])
            if sim > max_sim:
                max_sim = sim
                best_target = ref["target"]

        return best_target if max_sim >= self._similarity_threshold else self._generic_label

    def get_interaction_groups(self, mol: Chem.Mol) -> List[Dict[str, str]]:
        if not mol:
            return [{"name": "Unknown", "moa": "Unknown"}]

        matched = [
            {"name": ig["name"], "moa": ig["moa"]}
            for ig in self._interaction_groups
            if mol.HasSubstructMatch(ig["pattern"])
        ]
        return matched if matched else [{"name": "Non_Covalent", "moa": "Reversible_Inhibitor"}]

    def get_functional_prompts(self, mol: Chem.Mol) -> List[str]:
        if not mol:
            return []

        prompts: List[str] = []

        for func_name in (name for name in dir(Fragments) if name.startswith("fr_")):
            func = getattr(Fragments, func_name)
            try:
                if func(mol) > 0:
                    prompts.append(func_name.replace("fr_", ""))
            except Exception:
                pass

        for name, pattern in self._target_rules:
            if mol.HasSubstructMatch(pattern):
                prompts.append(name)

        return list(set(prompts))

    @classmethod
    def from_json(cls, config_path: str | Path) -> "ChemistryAnalyzer":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(config)


class KnowledgeGraphBuilder:
    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        if config_path:
            resolved = Path(config_path)
            self.analyzer = ChemistryAnalyzer.from_json(resolved)
            print(f"Loaded domain config: {resolved.name}")
        else:
            resolved = Path(CONFIG.get("KG_DOMAIN_CONFIG_PATH", resolve_domain_config_path(PROJECT_ROOT)))
            if not resolved.exists():
                resolved = resolve_domain_config_path(PROJECT_ROOT)
            self.analyzer = ChemistryAnalyzer.from_json(resolved)
            print(f"Loaded domain config: {resolved.name}")

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "KnowledgeGraphBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def nuke_and_prepare_db(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared old data")

            for constraint in session.run("SHOW CONSTRAINTS YIELD name").data():
                try:
                    session.run(f"DROP CONSTRAINT {constraint['name']}")
                except Exception:
                    pass

            for index in session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'").data():
                try:
                    session.run(f"DROP INDEX {index['name']}")
                except Exception:
                    pass

            print("Cleaned old schema")

            session.run("CREATE CONSTRAINT FOR (m:Molecule) REQUIRE m.smiles IS UNIQUE")
            session.run("CREATE INDEX FOR (s:Scaffold) ON (s.smiles)")
            session.run("CREATE INDEX FOR (t:Target) ON (t.name)")
            session.run("CREATE INDEX FOR (ig:Interaction_Group) ON (ig.name)")
            session.run("CREATE INDEX FOR (moa:MoA) ON (moa.name)")
            session.run("CREATE INDEX FOR (fp:FunctionalGroup) ON (fp.name)")
            print("Created new indexes")

    def import_batch(self, batch: List[Dict]) -> None:
        query = """
        UNWIND $batch AS row
        MERGE (m:Molecule {smiles: row.smiles})
        SET m.is_virtual = row.is_virtual,
            m.source     = row.source
        FOREACH (_ IN CASE WHEN row.label IS NOT NULL THEN [1] ELSE [] END |
            SET m.label = row.label
        )
        FOREACH (_ IN CASE WHEN row.is_virtual THEN [1] ELSE [] END |
            SET m.docking_affinity = row.docking_affinity,
                m.ligand_id        = row.ligand_id
        )
        MERGE (s:Scaffold {smiles: row.scaffold})
        MERGE (m)-[:HAS_SCAFFOLD]->(s)
        FOREACH (fp_name IN row.functional_prompts |
            MERGE (fp:FunctionalGroup {name: fp_name})
            MERGE (m)-[:HAS_FUNCTIONAL_GROUP]->(fp)
        )
        FOREACH (ig IN row.interaction_groups |
            MERGE (igNode:Interaction_Group {name: ig.name})
            MERGE (moa:MoA {name: ig.moa})
            MERGE (igNode)-[:ACTS_VIA]->(moa)
            MERGE (m)-[:HAS_INTERACTION_GROUP]->(igNode)
        )
        MERGE (t:Target {name: row.target})
        MERGE (m)-[:TESTED_AGAINST]->(t)
        """
        with self.driver.session() as session:
            session.run(query, batch=batch)

    def process_experimental_molecules(self, df: pd.DataFrame, smiles_col: str, label_col: str) -> None:
        print(f"Processing {len(df)} EXPERIMENTAL molecules...")
        batch_data: List[Dict] = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="KG experimental import", unit="mol"):
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            batch_data.append(
                {
                    "smiles": smiles,
                    "is_virtual": False,
                    "source": "Experimental",
                    "label": str(row[label_col]) if pd.notna(row[label_col]) else None,
                    "docking_affinity": None,
                    "ligand_id": None,
                    "scaffold": get_scaffold(mol),
                    "interaction_groups": self.analyzer.get_interaction_groups(mol),
                    "functional_prompts": self.analyzer.get_functional_prompts(mol),
                    "target": self.analyzer.assign_target(smiles),
                }
            )
            if len(batch_data) >= BATCH_SIZE:
                self.import_batch(batch_data)
                print(f"Imported experimental: {idx + 1}/{len(df)}")
                batch_data = []
        if batch_data:
            self.import_batch(batch_data)
        print(f"Completed {len(df)} experimental molecules")

    def process_denovo_molecules(self, df: pd.DataFrame) -> None:
        print(f"Processing {len(df)} DE NOVO molecules...")
        batch_data: List[Dict] = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="KG de novo import", unit="mol"):
            smiles = row["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            batch_data.append(
                {
                    "smiles": smiles,
                    "is_virtual": True,
                    "source": "DiffSBDD",
                    "label": None,
                    "docking_affinity": float(row["affinity"]),
                    "ligand_id": int(row["ligand_id"]),
                    "scaffold": get_scaffold(mol),
                    "interaction_groups": self.analyzer.get_interaction_groups(mol),
                    "functional_prompts": self.analyzer.get_functional_prompts(mol),
                    "target": self.analyzer.assign_target(smiles),
                }
            )
            if len(batch_data) >= BATCH_SIZE:
                self.import_batch(batch_data)
                print(f"Imported de novo: {idx + 1}/{len(df)}")
                batch_data = []
        if batch_data:
            self.import_batch(batch_data)
        print(f"Completed {len(df)} de novo molecules")


def main() -> None:
    print("\n" + "=" * 80)
    print("BUILDING EGFR KNOWLEDGE GRAPH")
    print("=" * 80)
    print(f"Experimental CSV: {EXPERIMENTAL_CSV}")
    print("Build policy: train split only (experimental data), de novo skipped")

    def _detect_smiles_column(df: pd.DataFrame) -> str:
        cols_upper = {c.upper(): c for c in df.columns}
        for candidate in ["SMILES", "SELFIES", "CANONICAL_SMILES", "SMILE"]:
            if candidate in cols_upper:
                return cols_upper[candidate]
        return df.columns[0]

    with KnowledgeGraphBuilder() as builder:
        builder.nuke_and_prepare_db()
        print("\n" + "=" * 80)
        print("STEP 1: IMPORT TRAIN DATA AFTER SCAFFOLD SPLIT")
        print("=" * 80)
        df_experimental = pd.read_csv(EXPERIMENTAL_CSV)
        smiles_col = _detect_smiles_column(df_experimental)
        label_col = CONFIG.get("LABEL_COLUMN", "Label")
        train_df, _ = stratified_scaffold_split(
            df=df_experimental,
            smiles_col=smiles_col,
            label_col=label_col,
            test_size=CONFIG.get("TEST_SIZE", 0.2),
            seed=CONFIG.get("RANDOM_STATE", 42),
        )
        print(f"Using train split only: {len(train_df)} molecules")
        builder.process_experimental_molecules(train_df, smiles_col=smiles_col, label_col=label_col)

    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH BUILD COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
