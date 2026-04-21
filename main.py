from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from config import CONFIG
from utils import detect_smiles_column, stratified_scaffold_split


def _label_to_int(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.lower().str.strip()
    return values.map(lambda x: 1 if x in {"1", "active", "true", "yes"} else 0).to_numpy(dtype=np.int64)


def _load_and_split() -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    data_csv = Path(CONFIG["DATA_CSV"])
    df = pd.read_csv(data_csv)
    smiles_col = detect_smiles_column(df)
    label_col = CONFIG["LABEL_COLUMN"]
    if label_col not in df.columns:
        raise KeyError(f"Missing label column `{label_col}` in `{data_csv}`")
    train_df, valid_df = stratified_scaffold_split(
        df=df,
        smiles_col=smiles_col,
        label_col=label_col,
        test_size=CONFIG["TEST_SIZE"],
        seed=CONFIG["RANDOM_STATE"],
    )
    return train_df, valid_df, smiles_col, label_col


def _build_bert_embeddings(train_df: pd.DataFrame, valid_df: pd.DataFrame, smiles_col: str) -> Tuple[np.ndarray, np.ndarray]:
    model_path = CONFIG["MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    batch_size = CONFIG["BERT_BATCH_SIZE"]
    max_length = CONFIG["BERT_MAX_LENGTH"]

    def encode(smiles_list: list[str]) -> np.ndarray:
        embs = []
        total_batches = (len(smiles_list) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(smiles_list), batch_size),
            total=total_batches,
            desc="BERT embedding batches",
            unit="batch",
        ):
            batch = smiles_list[i : i + batch_size]
            tokenized = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                tokenized = {k: v.to("cuda") for k, v in tokenized.items()}
            with torch.no_grad():
                out = model(**tokenized).last_hidden_state
                mask = tokenized["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            embs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embs)

    return encode(train_df[smiles_col].astype(str).tolist()), encode(valid_df[smiles_col].astype(str).tolist())


def run_buildkg() -> None:
    _ensure_neo4j_ready()
    from kg.build_graph import KnowledgeGraphBuilder

    overall = tqdm(total=4, desc="buildkg pipeline", unit="step")
    train_df, _, smiles_col, label_col = _load_and_split()
    overall.update(1)
    with KnowledgeGraphBuilder(config_path=CONFIG["DOMAIN_CONFIG_PATH"]) as builder:
        builder.nuke_and_prepare_db()
        overall.update(1)
        builder.process_experimental_molecules(train_df, smiles_col=smiles_col, label_col=label_col)
        overall.update(1)
        denovo_path = Path(CONFIG["DENOVO_CSV"])
        if denovo_path.exists():
            denovo_df = pd.read_csv(denovo_path)
            if not denovo_df.empty:
                builder.process_denovo_molecules(denovo_df)
    overall.update(1)
    overall.close()
    print("buildkg completed: KG train+DeNovo uploaded to Neo4j Docker.")


def run_extraction() -> None:
    _ensure_neo4j_ready()
    from kg.kg_encoder import (
        build_kg_embeddings_for_df,
        ensure_kg_embeddings,
        infer_test_embedding_with_injection,
    )
    embeddings_dir = Path(CONFIG["EMBEDDINGS_DIR"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    overall = tqdm(total=5, desc="extraction pipeline", unit="step")
    train_df, valid_df, smiles_col, label_col = _load_and_split()
    overall.update(1)
    _ = _label_to_int(train_df[label_col])
    _ = _label_to_int(valid_df[label_col])

    print("Extracting BERT embeddings...")
    bert_train, bert_valid = _build_bert_embeddings(train_df, valid_df, smiles_col=smiles_col)
    np.save(embeddings_dir / "bert_train.npy", bert_train)
    np.save(embeddings_dir / "bert_valid.npy", bert_valid)
    overall.update(1)

    print("Extracting KG train embeddings...")
    ensure_kg_embeddings()
    kg_train = build_kg_embeddings_for_df(train_df, smiles_col=smiles_col)
    np.save(embeddings_dir / "kg_train.npy", kg_train)
    overall.update(1)

    print("Extracting KG valid embeddings by inject/remove each molecule...")
    kg_valid_rows = []
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="KG valid inject/remove", unit="mol"):
        emb = infer_test_embedding_with_injection(smiles=str(row[smiles_col]))
        kg_valid_rows.append(emb)
    kg_valid = np.vstack(kg_valid_rows).astype(np.float32)
    np.save(embeddings_dir / "kg_valid.npy", kg_valid)
    overall.update(1)
    overall.update(1)
    overall.close()
    print(f"Extraction completed. Embeddings saved in: {embeddings_dir}")


def run_benchmark() -> None:
    from evaluation.benchmark import run_benchmark_from_embeddings

    embeddings_dir = Path(CONFIG["EMBEDDINGS_DIR"])
    overall = tqdm(total=2, desc="benchmark pipeline", unit="step")
    overall.update(1)
    run_benchmark_from_embeddings(embeddings_dir)
    overall.update(1)
    overall.close()
    print("Benchmark completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["buildkg", "extraction", "benchmark"], required=True)
    args = parser.parse_args()

    if args.mode == "buildkg":
        run_buildkg()
    elif args.mode == "extraction":
        run_extraction()
    elif args.mode == "benchmark":
        run_benchmark()


def _try_start_neo4j_docker() -> None:
    compose_file = Path(CONFIG["PROJECT_ROOT"]) / "docker-compose.yml"
    if not compose_file.exists():
        return
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "neo4j"],
            cwd=str(compose_file.parent),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def _ensure_neo4j_ready(max_wait_sec: int = 45) -> None:
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency `neo4j`. Install it in your active env with `pip install neo4j`."
        ) from exc

    uri = CONFIG["NEO4J_URI"]
    user = CONFIG["NEO4J_USER"]
    password = CONFIG["NEO4J_PASSWORD"]

    _try_start_neo4j_docker()
    deadline = time.time() + max_wait_sec
    last_error = None

    while time.time() < deadline:
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            print(f"Neo4j is ready at {uri}")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(3)
        finally:
            if driver is not None:
                driver.close()

    raise RuntimeError(
        "Cannot connect to Neo4j. "
        f"Current config: NEO4J_URI={uri}, NEO4J_USER={user}. "
        "Please start container `docker compose up -d neo4j` and verify password."
    ) from last_error


if __name__ == "__main__":
    main()
