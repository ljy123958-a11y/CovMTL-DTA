# CovMTL-DTA: Enhanced Multi-Task Learning for Covalent Inhibitor

This repository contains the implementation of **CovMTL-DTA**, an **Enhanced Multi-Task Learning Model** designed for predicting drug-target binding affinity (pIC50), specifically optimized for covalent inhibitors with SMARTS warhead features.

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
├── analyze_warheads.py          # Utility tool to generate warhead SMARTS for new molecules
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

### 1. Train the Model

To train the multi-task model with the provided dataset (which already includes pre-calculated SMARTS):

```bash
python train.py
```

The script will:
- Load the dataset and protein embeddings.
- Perform 5-fold cross-validation.
- Save model weights, predictions, and analysis reports to the `results/` directory.

### 2. Analyze Warheads (For New Data)

The provided dataset (`data/dataset.csv`) already contains the `warhead_smarts` column required by the model. 

However, if you want to use this model on **new molecules** that don't have SMARTS annotations, you can use `analyze_warheads.py` to generate the necessary SMARTS string for the model input.

```bash
python analyze_warheads.py
```

This tool allows you to:
- Input a **SMILES** string.
- Input a target **Residue** (e.g., `CYS`, `SER`).
- Output the matching **SMARTS pattern** (which corresponds to the `warhead_smarts` feature used in `train.py`).

**Example Interaction:**
```text
[1/2] 请输入 SMILES 字符串: 
> C=CC(=O)Nc1ccccc1

[2/2] 请输入目标残基名称 (例如 CYS, SER, 或留空跳过): 
> CYS

[分析结果]
...
     匹配数量: 1 个SMARTS模式
       1) [C,c]=[C,c]-[C,c,S,s]=[O]  <-- This is the SMARTS string needed for model input
```

## Results

Detailed performance metrics and analysis plots can be found in the `results/` directory after running the training script.

- **Overall Pearson R**: 0.7721
- **Overall RMSE**: 0.8180

## Citation

If you use this code in your research, please cite:
[Add your citation here]
