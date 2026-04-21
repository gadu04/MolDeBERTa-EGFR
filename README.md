# 🧬 EGFR Inhibitory Activity Prediction: Solving the Scaffold Hopping Problem with MolDeBERTa and Knowledge Graphs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_Database-008CC1)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 1. Abstract

Chemical language models (Foundation Models) such as **MolDeBERTa** have achieved State-of-the-Art (SOTA) performance in molecular property prediction. However, they face a severe limitation when tackling the **Scaffold Hopping** problem, i.e., the ability to accurately predict the activity of compounds with entirely novel structural backbones (Low/Very Low structural similarity compared to the training set).

This project proposes a hybrid architecture to overcome this limitation by integrating a **Biomedical Knowledge Graph (KG)**. By employing a Meta-Learner to combine prediction probabilities from MolDeBERTa with statistical graph features (inferred via K-NN Tanimoto similarity for unseen data), the system targets improved ROC-AUC specifically within low-similarity chemical spaces.

## 🏗️ 2. System Architecture

The system follows **zero data leakage** principles, uses **Scaffold Splitting**, and runs through three independent branches:

1. **The NLP Branch (MolDeBERTa):**
   End-to-end fine-tuning of `MolDeBERTa-base-123M-mtr` to extract sequence-based chemical signals directly from SMILES strings.

2. **The Graph Branch (Knowledge Graph):**
   - Queries biomedical statistical features (e.g., warhead count, mechanism-of-action count) from Neo4j.
   - Solves the validation "orphan node" problem via **K-NN Graph Projection (Neighborhood Aggregation)**, inferring graph features by Top-K Tanimoto similarity from the training set.

3. **The Ensemble Branch (Meta-Learner):**
   Uses Logistic Regression stacking to learn optimal soft-voting weights from MolDeBERTa and KG probabilities, then applies **Dynamic Thresholding** to maximize F1-Macro on imbalanced biomedical labels.

**Evaluation Baseline:** MolFormer (`ibm/MoLFormer-XL-both-10pct`) is integrated as a SOTA reference branch.

## ⚙️ 3. System Requirements & Installation

This project requires substantial compute for >100M parameter fine-tuning.

**Hardware Recommendations**
- GPU: NVIDIA RTX 3090 (24GB VRAM) or equivalent.
- Architecture: Ampere or newer (TF32/FP16 support recommended).

**Environment Setup (Conda + Pip)**
```bash
conda create -n Mol python=3.10 -y
conda activate Mol

# PyTorch CUDA 12.1 (recommended for RTX 30-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install transformers accelerate datasets rdkit xgboost scikit-learn neo4j matplotlib seaborn tqdm pandas numpy
```

**Graph Database**
- Initialize Neo4j via Docker/Desktop.
- Ensure connection is available at `bolt://localhost:7688` (configured in `config.py`).

## 🚀 4. Pipeline Execution

To reproduce results consistently, run modes in the exact order below.

### Step 1: Initialize the Knowledge Graph
```bash
python main.py --mode buildkg
```
Executes scaffold splitting, resets old graph state, creates required structures, and imports Train/DeNovo molecules into Neo4j.

### Step 2: Train the SOTA Baseline (MolFormer)
```bash
python main.py --mode molformer_training
```
Runs in strict FP32 precision (stability on Ampere), saves predictions to `output/predictions/`.

### Step 3: Fine-Tune the Core NLP Model (MolDeBERTa)
```bash
python main.py --mode finetuning
```

### Step 4: Train the Graph Branch (KG)
```bash
python main.py --mode kg_training
```
Extracts Cypher statistical features, performs K-NN Tanimoto inference for validation molecules, and trains an independent XGBoost classifier.

### Step 5: Ensemble and Evaluation (Benchmark)
```bash
python main.py --mode benchmark
```
Runs the Meta-Learner, performs dynamic threshold search, and generates the full benchmark report with plots.

## 📊 5. Key Results (Error Analysis)

The strength of this hybrid design is observed through Tanimoto-bin analysis:

- In the **High Similarity** region (`>0.8`), standalone MolDeBERTa already performs strongly.
- In **Very Low / Low Similarity** regions (`<0.6`), MolDeBERTa performance declines due to structural novelty (Scaffold Hopping barrier).
- In these difficult regions, KG-derived domain knowledge acts as a complementary signal, and the combined pipeline (`MolDeBERTa + KG`) is expected to improve robustness over a single NLP branch.

**Generated figures are saved in `output/plots/`:**
- `auc_comparison_detailed.png`
- `active_prediction_counts.png`
- `heatmap_detailed.png`
- `improvement_detailed.png`
- `model_metrics_compairison.png`
- `roc_comparison.png`

---

### Practical Notes

- Current KG feature files are generated from:
  - `output/predictions/kg_features_train.csv`
  - `output/predictions/kg_features_valid_inferred.csv`
- Constant/near-constant KG columns are automatically removed before KG model training.
- Benchmark automatically loads MolFormer predictions if available (or triggers generation when configured).