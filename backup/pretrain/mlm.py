from datasets import load_dataset
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import numpy as np
import os
import click

VALID_MODEL_SIZE = ['tiny', 'small', 'base']
VALID_DATASETS = ['10M', '123M']

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
  if dataset == '10M':
    dataset_dict = load_dataset("SaeedLab/MolDeBERTa", data_dir="10M")
  else:
    dataset_dict = load_dataset("SaeedLab/MolDeBERTa", data_dir="123M")
  
  dataset_dict = dataset_dict['train']
  split_dataset = dataset_dict.train_test_split(test_size=0.01, seed=42)
  train_dataset = split_dataset["train"]
  valid_dataset = split_dataset["test"]

  tokenizer = AutoTokenizer.from_pretrained("../moldeberta-tokenizer", use_fast=True)

  def tokenize_smiles(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

  train_dataset = train_dataset.map(tokenize_smiles, batched=True, num_proc=4, remove_columns=["text"])
  valid_dataset = valid_dataset.map(tokenize_smiles, batched=True, num_proc=4, remove_columns=["text"])

  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
  )

  if model_size == 'tiny':
    hidden_size = 384
    num_hidden_layers = 3
    num_attention_heads = 6
    intermediate_size = 1536
    per_device_train_batch_size = 2048
    learning_rate = 1e-4
  
  elif model_size == 'small':
    hidden_size = 512
    num_hidden_layers = 6
    num_attention_heads = 8
    intermediate_size = 2048
    per_device_train_batch_size = 1024
    learning_rate = 5e-5

  else:
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    per_device_train_batch_size = 512
    learning_rate = 5e-5

  config = DebertaV2Config(
    vocab_size=len(tokenizer),
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    max_position_embeddings=128,
  )

  model = DebertaV2ForMaskedLM(config)

  output_dir = f"../pretrained/moldeberta-{model_size}-{dataset}-mlm"

  training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    learning_rate=learning_rate,
    weight_decay=0.01,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    logging_steps=1000,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
  )

  trainer.train()
  trainer.save_model(output_dir)
  tokenizer.save_pretrained(output_dir)

  os.system(f"rm {output_dir}/checkpoint* -r")

if __name__ == '__main__':
  main()