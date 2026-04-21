import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from captum.attr import GradientShap
import matplotlib.cm as cm
import numpy as np
import re
import click
import os

class MolecularExplainer:
  def __init__(self, model, tokenizer, device='cuda'):
    self.model = model.to(device)
    self.model.eval()
    self.device = device
    self.tokenizer = tokenizer
    self.embedding_layer = self.get_embedding_layer()

  def get_embedding_layer(self):
    for name, module in self.model.named_modules():
      if re.search("embeddings", name) and hasattr(module, "word_embeddings"):
        return module.word_embeddings
    for name, module in self.model.named_modules():
        if re.search("embed_tokens", name):
            return module
    raise ValueError("Embedding layer not found")

  def tokenize_smiles(self, smiles):
    inputs = self.tokenizer(smiles, return_tensors="pt").to(self.device)
    decoded_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    not_special_tokens = [t not in self.tokenizer.all_special_tokens for t in decoded_tokens]
    decoded_tokens = [t for t in decoded_tokens if t not in self.tokenizer.all_special_tokens]
    return inputs, decoded_tokens, not_special_tokens

  def forward_for_captum(self, inputs_embeds, attention_mask):
    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return outputs.logits

  def compute_atom_importance(self, smiles, target_label=0, n_samples=50):
    inputs, decoded_tokens, not_special_tokens = self.tokenize_smiles(smiles)
    
    embeddings = self.embedding_layer(inputs['input_ids'])
    baseline = torch.zeros_like(embeddings)

    gs = GradientShap(self.forward_for_captum)

    attr = gs.attribute(
      inputs=embeddings,
      baselines=baseline,
      target=target_label,
      additional_forward_args=(inputs["attention_mask"],),
      n_samples=n_samples
    )

    token_importance = attr.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    token_importance = token_importance[not_special_tokens]
    return token_importance, decoded_tokens

  def map_atoms_to_tokens(self, mol, decoded_tokens):
    atom_to_tokens = {a.GetIdx(): [] for a in mol.GetAtoms()}
    atom_idx = 0
    for tok_id, tok in enumerate(decoded_tokens):
      matches = re.findall(r'(?:Br|Cl|Rb|Mg|Kr|Xe|Zn|Se|Ba|Cs|Li|Na|He|Al|Sr|Ga|Ra|Ca|Be|As|Ag|Bi|Te|Si|[B-Z][a-z]?|[bcnops])', tok)
      for m in matches:
        if atom_idx < mol.GetNumAtoms():
          atom_to_tokens[atom_idx].append(tok_id)
          atom_idx += 1
    return atom_to_tokens

  def explain(self, smiles, target_label=0, n_samples=50, filename='../explainer_images/molecule.png'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      raise ValueError("Invalid SMILES")

    token_importance, decoded_tokens = self.compute_atom_importance(smiles, target_label, n_samples)
    atom_to_tokens = self.map_atoms_to_tokens(mol, decoded_tokens)

    atom_importance_per_atom = np.zeros(mol.GetNumAtoms())
    for atom_idx, token_ids in atom_to_tokens.items():
      atom_importance_per_atom[atom_idx] = token_importance[token_ids].sum()

    aimp = (atom_importance_per_atom - atom_importance_per_atom.min()) / (
        atom_importance_per_atom.max() - atom_importance_per_atom.min() + 1e-9
    )

    cmap_atoms = cm.get_cmap('coolwarm')
    atom_colors = {i: tuple(cmap_atoms(float(aimp[i]))[:3]) for i in range(mol.GetNumAtoms())}

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightAtomRadii={i: 0.4 for i in atom_colors.keys()}
    )
    drawer.FinishDrawing()
    drawer.WriteDrawingText(filename)

@click.command()
@click.option('--model_path',
  type=str,
  required=True,
  help="Model path"
)
@click.option('--smiles',
  type=str,
  required=True,
  help="The SMILES string"
)
@click.option('--target_label',
  type=int,
  required=True,
  help="The target label"
)
def main(model_path, smiles, target_label):
  os.makedirs('../explainer_images', exist_ok=True)
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  explainer = MolecularExplainer(model=model, tokenizer=tokenizer)
  explainer.explain(smiles, target_label=target_label)

if __name__ == '__main__':
  main()