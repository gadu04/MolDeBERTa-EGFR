from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_domain_config_path(project_root: Path) -> Path:
    return project_root / "domain_config.json"


def _env(key: str, default: Any) -> Any:
    return os.getenv(key, default)


CONFIG: Dict[str, Any] = {
    "PROJECT_ROOT": str(PROJECT_ROOT),
    "DATA_CSV": str(PROJECT_ROOT / "data" / "data_end.csv"),
    "DENOVO_CSV": str(PROJECT_ROOT / "data" / "DeNovo_Molecule.csv"),
    "OUTPUT_DIR": str(PROJECT_ROOT / "output"),
    "EMBEDDINGS_DIR": str(PROJECT_ROOT / "output" / "embeddings"),
    "EVAL_DIR": str(PROJECT_ROOT / "output" / "benchmark"),
    "PLOT_DIR": str(PROJECT_ROOT / "output" / "plots"),
    "MODEL_PATH": str(PROJECT_ROOT / "model"),
    "MOLFORMER_MODEL_PATH": _env("MOLFORMER_MODEL_PATH", "ibm/MoLFormer-XL-both-10pct"),
    "DOMAIN_CONFIG_PATH": str(resolve_domain_config_path(PROJECT_ROOT)),
    "SMILES_COLUMN": _env("SMILES_COLUMN", "SMILES"),
    "LABEL_COLUMN": _env("LABEL_COLUMN", "Label"),
    "TEST_SIZE": float(_env("TEST_SIZE", 0.2)),
    "VALID_SIZE": float(_env("VALID_SIZE", 0.2)),
    "RANDOM_STATE": int(_env("RANDOM_STATE", 42)),
    "BERT_MAX_LENGTH": int(_env("BERT_MAX_LENGTH", 128)),
    "BERT_BATCH_SIZE": int(_env("BERT_BATCH_SIZE", 32)),
    "KG_EMBED_DIM": int(_env("KG_EMBED_DIM", 128)),
    "KG_BUILD_BATCH_SIZE": int(_env("KG_BUILD_BATCH_SIZE", 1000)),
    "WEIGHT_SEARCH_STEP": float(_env("WEIGHT_SEARCH_STEP", 0.1)),
    "TANIMOTO_BINS": [
        (0.0, 0.2, "Very Low"),
        (0.2, 0.4, "Low"),
        (0.4, 0.7, "Medium"),
        (0.7, 1.0, "High"),
    ],
    "NEO4J_URI": _env("NEO4J_URI", "bolt://localhost:7688"),
    "NEO4J_USER": _env("NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": _env("NEO4J_PASSWORD", "12345678"),
}
