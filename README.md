# Linear Algebra Based Discovery of Disease-Linked Gene Expression Patterns

## Overview

This project applies linear algebra techniques to discover latent patterns in high-dimensional gene expression data that correlate with disease states. The core mathematical insight is that **disease states correspond to specific subspaces** within the gene expression space that can be discovered through matrix decomposition.

## Mathematical Foundation

Gene expression data is represented as a matrix **X** ∈ ℝ^(m×n) where:
- *m* = number of genes (typically 10,000-50,000)
- *n* = number of samples (typically 100-1,000)

### Core Techniques

1. **Singular Value Decomposition (SVD)**: X = U Σ V^T
2. **Principal Component Analysis (PCA)**: Eigenvalue decomposition of covariance matrix
3. **Non-negative Matrix Factorization (NMF)**: X ≈ WH (parts-based representation)
4. **Eigenvalue Analysis**: Gene-gene correlation network analysis

## Project Structure

```
gene-expression-la-analysis/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Initial data inspection
│   ├── 02_preprocessing.ipynb         # Normalization, QC
│   ├── 03_svd_pca_analysis.ipynb      # Linear decomposition
│   ├── 04_nmf_metagenes.ipynb         # Non-negative factorization
│   ├── 05_correlation_networks.ipynb  # Eigenvalue analysis
│   ├── 06_disease_classification.ipynb # Pattern validation
│   └── 07_biomarker_discovery.ipynb   # Gene selection
├── src/
│   ├── preprocessing.py               # Reusable preprocessing functions
│   ├── linear_algebra.py              # LA utility functions
│   ├── visualization.py               # Plotting utilities
│   └── validation.py                  # Statistical validation
├── data/
│   ├── raw/                           # Original downloaded data
│   ├── processed/                     # Cleaned expression matrices
│   └── results/                       # Output figures, tables
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Download Data**: Place your Kaggle dataset in `data/raw/`
2. **Run Notebooks**: Execute notebooks in order (01 → 07)
3. **Results**: Find outputs in `data/results/`

## Recommended Datasets

| Dataset | Genes | Samples | Disease Types |
|---------|-------|---------|---------------|
| TCGA Pan-Cancer RNA-seq | ~20,000 | ~10,000 | 33 cancer types |
| Breast Cancer Gene Expression | ~15,000 | ~800 | ER+/ER-, Subtypes |
| GTEx Normal Tissue Expression | ~56,000 | ~17,000 | Multiple tissues |
| Leukemia Gene Expression (Golub) | 7,129 | 72 | ALL/AML |

## Key Outputs

- **Dimensionality Estimate**: Optimal k for low-rank approximation
- **Metagene Matrix**: Co-expressed gene modules
- **Gene Rankings**: Scores for disease-discriminating genes
- **Classification Accuracy**: Cross-validated performances
- **Biomarker Panels**: Minimal gene sets for disease classification

## Validation Metrics

| Metric | Threshold |
|--------|-----------|
| Reconstruction R² | > 0.70 |
| Component Stability | > 0.85 |
| Classification AUC | > 0.80 |
| Permutation Test p-value | < 0.01 |
