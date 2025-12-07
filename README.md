# Enhanced Multi-Task Learning for covalent inhibitor

This repository contains the implementation of an **Enhanced Multi-Task Learning Model** designed for predicting drug-target binding affinity (pIC50), specifically optimized for covalent inhibitors with SMARTS warhead features.

## Key Features

- **Task Relation Module**: Automatically learns relationships between 54 different protein targets (tasks) without requiring manual protein family annotation.
- **Enhanced Cross-Modal Attention**: Uses multi-head attention with a gated fusion mechanism to better integrate molecular and protein features.
- **SMARTS Warhead Features**: Incorporates reactive warhead information specifically for covalent inhibitors.
- **Performance**: Achieved Pearson correlation of **0.77** across all tasks in 5-fold cross-validation.

## Project Structure

```
├── data/
│   └── dataset.csv              # Primary dataset with SMILES, Target_Gene, pIC50, and Warhead SMARTS
├── results/                     # Output directory for training logs and predictions
├── protein_embeddings.pkl       # Pre-computed ESM protein embeddings
├── train.py                     # Main training script
└── README.md                    # This file
```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- RDKit
- ESM (Evolutionary Scale Modeling)
- pandas, numpy, scipy, matplotlib

## Usage

1. **Install Dependencies**:
   Ensure you have all required libraries installed.

2. **Prepare Data**:
   The `data/dataset.csv` should contain the following columns:
   - `SMILES`: Molecular structure
   - `Target_Gene`: Target protein name
   - `pic50`: Binding affinity label
   - `warhead_smarts`: SMARTS string for the covalent warhead
   - `proteinseq`: Amino acid sequence of the target

3. **Run Training**:
   ```bash
   python train.py
   ```
   The script will:
   - Load the dataset and protein embeddings.
   - Perform 5-fold cross-validation.
   - Save model weights, predictions, and analysis reports to the `results/` directory.

## Results

Detailed performance metrics and analysis plots can be found in the `results/` directory after running the training script.

- **Overall Pearson R**: 0.7721
- **Overall RMSE**: 0.8180

## Citation

If you use this code in your research, please cite:
[Add your citation here]

