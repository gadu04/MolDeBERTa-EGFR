import torch
import torch.nn as nn
import deepchem as dc
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import click
import random
import os

VALID_TASKS = ['bace-classification', 'bace-regression', 'bbbp', 'clearance', 'clintox', 'delaney', 'hiv', 'lipo', 'tox21']

tokenizer = None
model_name = None
y_mean = None
y_std = None

task_type = {
  'bace-classification': 'classification',
  'bace-regression': 'regression',
  'bbbp': 'classification',
  'clearance': 'regression',
  'clintox': 'classification',
  'delaney': 'regression',
  'hiv': 'classification',
  'lipo': 'regression',
  'tox21': 'classification'
}

@click.command()
@click.option('--model_path')
@click.option('--task',
  type=click.Choice(VALID_TASKS, case_sensitive=True),
  required=True,
  help="The task for finetuning"
)

def main(model_path, task):
  global tokenizer, model_name, y_mean, y_std
  model_name = model_path
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  out_dir_model_path = model_path.replace('../', '').replace('/', '_')

  df_train, df_valid, df_test = load_dataset(task)

  if task_type[task] == 'regression':
    y_mean = df_train["label"].mean()
    y_std = df_train["label"].std()
    df_train["label"] = (df_train["label"] - y_mean) / y_std
    df_valid["label"] = (df_valid["label"] - y_mean) / y_std
    df_test["label"]  = (df_test["label"] - y_mean) / y_std

  else:
    cw = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(df_train.label.values),
        y = df_train.label.values
      )
    cw = dict(zip([0, 1], cw))
    cw = torch.tensor(list(cw.values()), dtype=torch.float)

  train_hf = Dataset.from_pandas(df_train)
  valid_hf = Dataset.from_pandas(df_valid)
  test_hf  = Dataset.from_pandas(df_test)

  train_hf = train_hf.map(tokenize, batched=True)
  valid_hf = valid_hf.map(tokenize, batched=True)
  test_hf  = test_hf.map(tokenize, batched=True)

  if task_type[task] == 'regression':
    training_args = TrainingArguments(
      output_dir=f"../finetuned/{out_dir_model_path}-{task}",
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=1,
      learning_rate=5e-5,
      per_device_train_batch_size=32,
      per_device_eval_batch_size=32,
      num_train_epochs=100,
      load_best_model_at_end=True,
      metric_for_best_model="rmse",
      greater_is_better=False
    )

    trainer = RegressionTrainer(
      model=None,
      args=training_args,
      train_dataset=train_hf,
      eval_dataset=valid_hf,
      compute_metrics=compute_metrics_regression,
      model_init=model_init_regression,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    best_run = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    n_trials=20,
    hp_space=lambda _: {
      "learning_rate": np.random.uniform(1e-5, 5e-4),
      "per_device_train_batch_size": np.random.choice([16, 32, 64]),
      "seed": np.random.randint(1, 10000)
      }
    )

  else:
    training_args = TrainingArguments(
      output_dir=f"../finetuned/{out_dir_model_path}-{task}",
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=1,
      learning_rate=5e-5,
      per_device_train_batch_size=32,
      per_device_eval_batch_size=32,
      num_train_epochs=100,
      load_best_model_at_end=True,
      metric_for_best_model="roc_auc",
      greater_is_better=True
    )

    trainer = WeightedLossTrainer(
      model=None,
      args=training_args,
      train_dataset=train_hf,
      eval_dataset=valid_hf,
      compute_metrics=compute_metrics_classification,
      class_weights=cw,
      model_init=model_init_classification,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    best_run = trainer.hyperparameter_search(
      direction="maximize",
      backend="optuna",
      n_trials=20,
      hp_space=lambda _: {
        "learning_rate": np.random.uniform(1e-5, 5e-4),
        "per_device_train_batch_size": np.random.choice([16, 32, 64]),
        "seed": np.random.randint(1, 10000)
      }
    )

  best_params = best_run.hyperparameters
  for key, value in best_params.items():
    if hasattr(trainer.args, key):
      setattr(trainer.args, key, value)

  trainer.train()
  metrics = trainer.evaluate(test_hf)
  print('Results on the test set:', metrics)
  trainer.save_model(f"../finetuned/{out_dir_model_path}-{task}-finetuned")
  tokenizer.save_pretrained(f"../finetuned/{out_dir_model_path}-{task}-finetuned")
  os.system(f'rm "../finetuned/{out_dir_model_path}-{task}" -r')

def to_dataframe(dataset, task):
  if task == 'clintox':
    return pd.DataFrame({
      "smiles": dataset.ids,
      "label": dataset.y[:, 1].flatten()
    })
  elif task == 'tox21':
    return pd.DataFrame({
      "smiles": dataset.ids,
      "label": dataset.y[:, 11].flatten()
    })
  else:
    return pd.DataFrame({
      "smiles": dataset.ids,
      "label": dataset.y.flatten()
    })

def load_dataset(task):
  if task == 'bace-classification':
    tasks, datasets, transformers = dc.molnet.load_bace_classification(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'bace-regression':
    tasks, datasets, transformers = dc.molnet.load_bace_regression(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'bbbp':
    tasks, datasets, transformers = dc.molnet.load_bbbp(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'clearance':
    tasks, datasets, transformers = dc.molnet.load_clearance(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'clintox':
    tasks, datasets, transformers = dc.molnet.load_clintox(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'delaney':
    tasks, datasets, transformers = dc.molnet.load_delaney(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'hiv':
    tasks, datasets, transformers = dc.molnet.load_hiv(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'lipo':
    tasks, datasets, transformers = dc.molnet.load_lipo(
      featurizer="Raw",
      splitter="scaffold"
    )
  elif task == 'tox21':
    tasks, datasets, transformers = dc.molnet.load_tox21(
      featurizer="Raw",
      splitter="scaffold"
    )

  train_dataset, valid_dataset, test_dataset = datasets
  df_train = to_dataframe(train_dataset, task)
  df_valid = to_dataframe(valid_dataset, task)
  df_test  = to_dataframe(test_dataset, task)

  return df_train, df_valid, df_test

def tokenize(batch):
  return tokenizer(batch["smiles"], padding="max_length", truncation=True, max_length=128)

def compute_metrics_regression(eval_pred):
  logits, labels = eval_pred
  logits = logits.squeeze()
  labels = labels.squeeze()
  preds = logits * y_std + y_mean
  labels = labels * y_std + y_mean
  rmse = mean_squared_error(labels, preds) ** 0.5
  return {"rmse": rmse}

class RegressionTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits").squeeze(-1)
    criterion = nn.MSELoss()
    loss = criterion(logits, labels.float())
    return (loss, outputs) if return_outputs else loss

def model_init_regression(trial):
  return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")

def compute_metrics_classification(eval_pred):
  logits, labels = eval_pred
  probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
  auc = roc_auc_score(labels, probs)
  return {"roc_auc": auc}

class WeightedLossTrainer(Trainer):
  def __init__(self, *args, class_weights, **kwargs):
    super().__init__(*args, **kwargs)
    self.class_weights = class_weights

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get('logits')
    criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
    loss = criterion(logits, labels.long())
    return (loss, outputs) if return_outputs else loss

def model_init_classification(trial):
  return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

if __name__ == '__main__':
  main()