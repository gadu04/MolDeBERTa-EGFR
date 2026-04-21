from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from datasets import load_dataset 
import os

os.makedirs("../moldeberta-tokenizer", exist_ok=True)

dataset_dict = load_dataset("SaeedLab/MolDeBERTa", data_dir="10M")
dataset = dataset_dict['train']

def batch_iterator(batch_size=10000):
  for i in range(0, len(dataset), batch_size):
    yield dataset[i:i+batch_size]["text"]

tokenizer_bpe = ByteLevelBPETokenizer()

tokenizer_bpe.train_from_iterator(
  batch_iterator(),
  vocab_size=4000,
  min_frequency=2,
  special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
)

tokenizer_bpe.save_model("../moldeberta-tokenizer")

tokenizer = RobertaTokenizerFast(
  vocab_file="../moldeberta-tokenizer/vocab.json",
  merges_file="../moldeberta-tokenizer/merges.txt",
  unk_token="[UNK]",
  pad_token="[PAD]",
  cls_token="[CLS]",
  sep_token="[SEP]",
  mask_token="[MASK]"
)

tokenizer.save_pretrained("../moldeberta-tokenizer")