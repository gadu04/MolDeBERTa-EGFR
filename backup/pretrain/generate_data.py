import os
import sys
import torch
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from transformers import AutoTokenizer
import click
from datasets import load_dataset

OUTPUT_DIR = "../chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHUNK_SIZE = 10000 
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) 
FP_BITS = 2048
FP_RADIUS = 2
MAX_LENGTH = 128

desc_names = [i[0] for i in Descriptors.descList]
desc_names.pop(42)
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
DESC_LEN = len(desc_names)

tokenizer = AutoTokenizer.from_pretrained("../moldeberta-tokenizer")

VALID_DATASETS = ['10M', '123M']

@click.command()
@click.option('--dataset',
  type=click.Choice(VALID_DATASETS, case_sensitive=True),
  required=True,
  help="The dataset (10M or 123M)"
)
def main(dataset):
  if dataset == '10M':
    dataset_dict = load_dataset("SaeedLab/MolDeBERTa", data_dir="10M")
  else:
    dataset_dict = load_dataset("SaeedLab/MolDeBERTa", data_dir="123M")
  
  dataset_dict = dataset_dict['train']

  chunk = []
  chunk_idx = 0
  for i, data in enumerate(dataset_dict):
    smi = data['text']
    chunk.append(smi)
    if len(chunk) >= CHUNK_SIZE:
      process_chunk(chunk, chunk_idx, dataset)
      chunk = []
      chunk_idx += 1

  if chunk:
    process_chunk(chunk, chunk_idx, dataset)

def process_molecule(smi):
  mol = Chem.MolFromSmiles(smi)
  desc_array = np.zeros(DESC_LEN, dtype=np.float32)
  fp_array = np.zeros(FP_BITS, dtype=np.uint8)
  
  if mol is None:
    return desc_array, fp_array

  try:
    d_vals = calculator.CalcDescriptors(mol)
    d_arr = np.array(d_vals, dtype=np.float64)
    d_arr = np.nan_to_num(d_arr, nan=0.0, posinf=0.0, neginf=0.0)
    max_f32 = np.finfo(np.float32).max
    min_f32 = np.finfo(np.float32).min
    d_arr = np.clip(d_arr, min_f32, max_f32)
    desc_array = d_arr.astype(np.float32)
  except:
    pass
  try:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
    fp_array = np.array(fp, dtype=np.uint8)
  except:
    pass

  return desc_array, fp_array

def process_chunk(chunk, chunk_idx, dataset):
  results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(process_molecule)(smi) for smi in chunk
  )
  
  descriptors_batch, fingerprints_batch = zip(*results)
  
  token_batch = tokenizer(
    chunk,
    truncation=True,
    padding='max_length',
    max_length=MAX_LENGTH,
    return_tensors='pt'
  )
  token_path = os.path.join(OUTPUT_DIR, f"tokens_{chunk_idx}_{dataset}.pt")
  torch.save(
    {"input_ids": token_batch["input_ids"], "attention_mask": token_batch["attention_mask"]},
    token_path
  )

  desc_path = os.path.join(OUTPUT_DIR, f"descriptors_{chunk_idx}_{dataset}.npy")
  fp_path = os.path.join(OUTPUT_DIR, f"fingerprints_{chunk_idx}_{dataset}.npy")
  
  np.save(desc_path, np.array(descriptors_batch, dtype=np.float32))
  np.save(fp_path, np.array(fingerprints_batch, dtype=np.uint8))

if __name__ == "__main__":
  main()