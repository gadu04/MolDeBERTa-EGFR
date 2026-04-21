import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import random, numpy as np
import os
import gc
from accelerate import Accelerator
import click

CHUNKED_DATA_PATH = '../chunks/tokens_{}_{}.pt'
CHUNKED_LABELS = '../chunks/descriptors_{}_{}.npy'
CHECKPOINT_PATH = '../pretrained/moldeberta-{}-{}-mtr'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
CHUNK_SIZE = 10000
VALID_MODEL_SIZE = ['tiny', 'small', 'base']
VALID_DATASETS = ['10M', '123M']

class SMILESDataset(Dataset):
  def __init__(self, chunk_indices, means, stds, dataset_name):
    self.chunk_indices = chunk_indices
    self.means = means
    self.stds = stds
    self.dataset_name = dataset_name

    self.current_chunk_idx = -1
    self.current_chunk_data = []
    self.current_chunk_label = []
    self.permutation = []

  def __len__(self):
    return len(self.chunk_indices) * CHUNK_SIZE

  def __getitem__(self, idx):
    chunk_id = idx // CHUNK_SIZE
    idx_in_chunk = idx % CHUNK_SIZE

    target_chunk = self.chunk_indices[chunk_id]

    if target_chunk != self.current_chunk_idx:
      chunk_path = CHUNKED_DATA_PATH.format(target_chunk, self.dataset_name)
      chunk_label_path = CHUNKED_LABELS.format(target_chunk, self.dataset_name)
      self.current_chunk_data = torch.load(chunk_path, map_location='cpu')
      self.current_chunk_label = np.load(chunk_label_path)
      self.current_chunk_idx = target_chunk
      self.permutation = torch.randperm(CHUNK_SIZE).tolist()

    real_id = self.permutation[idx_in_chunk]
    input_ids = self.current_chunk_data['input_ids'][real_id]
    attention_mask = self.current_chunk_data['attention_mask'][real_id]
    y = self.current_chunk_label[real_id]
    y = torch.tensor(y, dtype=torch.float)
    y = torch.clamp(y, -100, 100)
    means = torch.tensor(self.means, dtype=torch.float)
    stds = torch.tensor(self.stds, dtype=torch.float)
    epsilon = 1e-6
    y = (y - means) / (stds + epsilon)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.clamp(y, -10, 10)
    return input_ids, attention_mask, y

@click.command()
@click.option('--model_size',
  type=click.Choice(VALID_MODEL_SIZE, case_sensitive=True),
  required=True,
  help="Model size (tiny, small, or base)"
)
@click.option('--dataset',
  type=click.Choice(VALID_DATASETS, case_sensitive=True),
  required=True,
  help="The dataset (10M or 123M)"
)

def main(model_size, dataset):
  accelerator = Accelerator(mixed_precision="fp16")
  
  if model_size == 'tiny':
    hidden_size = 384
    num_hidden_layers = 3
    num_attention_heads = 6
    intermediate_size = 1536
    batch_size = 2048
    learning_rate = 1e-4
  
  elif model_size == 'small':
    hidden_size = 512
    num_hidden_layers = 6
    num_attention_heads = 8
    intermediate_size = 2048
    batch_size = 1024
    learning_rate = 5e-5

  else:
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    batch_size = 512
    learning_rate = 5e-5

  if dataset == '10M':
    total_chunks_available = 1000
  else:
    total_chunks_available = 12300

  all_chunks = list(range(total_chunks_available))
  random.shuffle(all_chunks)

  split_idx = int(len(all_chunks) * 0.99)
  train_chunks = all_chunks[:split_idx]
  val_chunks = all_chunks[split_idx:]

  running_sum = np.zeros(216, dtype=np.float64)
  running_sq_sum = np.zeros(216, dtype=np.float64)
  total_count = 0
    
  for chunk_id in tqdm(train_chunks, desc="Computing Stats"):
    labels = np.load(CHUNKED_LABELS.format(chunk_id, dataset))
    running_sum += np.nansum(labels, axis=0)
    running_sq_sum += np.nansum(labels ** 2, axis=0)
    total_count += labels.shape[0]

  means = running_sum / total_count
  variances = (running_sq_sum / total_count) - (means ** 2)
  variances = np.maximum(variances, 0)
  stds = np.sqrt(variances)

  means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
  stds = np.nan_to_num(stds, nan=1.0, posinf=1.0, neginf=1.0)
  stds[stds == 0] = 1.0

  train_dataset = SMILESDataset(train_chunks, means, stds, dataset)
  val_dataset = SMILESDataset(val_chunks, means, stds, dataset)

  train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    num_workers=8,
    pin_memory=True,
    shuffle=False 
  )
  val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    num_workers=8, 
    pin_memory=True,
    shuffle=False
  )

  tokenizer = AutoTokenizer.from_pretrained("../moldeberta-tokenizer", use_fast=True)

  config = DebertaV2Config(
    vocab_size=len(tokenizer),
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    max_position_embeddings=128,
    num_labels=216
  )

  model = DebertaV2ForSequenceClassification(config)

  optimizer = AdamW(model.parameters(), lr=learning_rate)
  criterion = nn.MSELoss()

  model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
  )

  for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    if accelerator.is_main_process:
      pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    else:
      pbar = train_loader

    for input_ids, attention_mask, y in pbar:
      optimizer.zero_grad(set_to_none=True)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      loss = criterion(logits, y)

      accelerator.backward(loss)
      optimizer.step()

      train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_loss = accelerator.gather(torch.tensor(avg_train_loss).to(accelerator.device)).mean().item()

    if accelerator.is_main_process:
      print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for input_ids, attention_mask, y in val_loader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, y)
        val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_loss = accelerator.gather(torch.tensor(avg_val_loss).to(accelerator.device)).mean().item()

    if accelerator.is_main_process:
      print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    accelerator.wait_for_everyone()
    gc.collect()

  accelerator.wait_for_everyone()
  if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    final_path = CHECKPOINT_PATH.format(model_size, dataset)  
    foundation_model = unwrapped_model.deberta
    foundation_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved at {final_path}")

if __name__ == '__main__':
  main()