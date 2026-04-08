"""
preprocessing.py
Reusable preprocessing functions for gene expression data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_geo_matrix(filepath: str) -> pd.DataFrame:
    """Load a GEO series matrix file into a DataFrame (genes × samples)."""
    df = pd.read_csv(filepath, sep='\t', comment='!', index_col=0)
    df = df.dropna(how='all')
    print(f"Loaded matrix: {df.shape[0]} probes × {df.shape[1]} samples")
    return df


def load_expression_csv(filepath: str, index_col: int = 0) -> pd.DataFrame:
    """Load a generic CSV expression matrix."""
    df = pd.read_csv(filepath, index_col=index_col)
    print(f"Loaded expression matrix: {df.shape[0]} genes × {df.shape[1]} samples")
    return df


def load_metadata(filepath: str, sample_col: str = None) -> pd.DataFrame:
    """Load sample metadata / phenotype table."""
    meta = pd.read_csv(filepath, index_col=sample_col)
    print(f"Metadata loaded: {meta.shape[0]} samples × {meta.shape[1]} columns")
    return meta


# ─────────────────────────────────────────────
# 2. QUALITY CONTROL
# ─────────────────────────────────────────────

def qc_report(expr: pd.DataFrame) -> dict:
    """
    Basic QC metrics for an expression matrix.
    Returns a dict with key stats and prints a summary.
    """
    n_genes, n_samples = expr.shape
    missing_pct = expr.isnull().values.mean() * 100
    zero_pct    = (expr == 0).values.mean() * 100
    per_gene_var = expr.var(axis=1)
    low_var_genes = (per_gene_var < per_gene_var.quantile(0.10)).sum()

    report = {
        'n_genes':       n_genes,
        'n_samples':     n_samples,
        'missing_pct':   round(missing_pct, 3),
        'zero_pct':      round(zero_pct, 3),
        'low_var_genes': low_var_genes,
        'mean_expr':     round(expr.values.mean(), 4),
        'median_expr':   round(np.nanmedian(expr.values), 4),
    }

    print("=" * 45)
    print("  QC REPORT")
    print("=" * 45)
    for k, v in report.items():
        print(f"  {k:<20}: {v}")
    print("=" * 45)
    return report


def filter_low_expression(expr: pd.DataFrame,
                           min_mean: float = 1.0,
                           min_expressed_frac: float = 0.2) -> pd.DataFrame:
    """
    Remove genes whose mean expression < min_mean OR that are expressed
    in fewer than min_expressed_frac of samples.
    """
    mean_filter = expr.mean(axis=1) >= min_mean
    frac_filter = (expr > 0).mean(axis=1) >= min_expressed_frac
    keep = mean_filter & frac_filter
    filtered = expr.loc[keep]
    print(f"Low-expression filter: {expr.shape[0]} → {filtered.shape[0]} genes retained")
    return filtered


def filter_low_variance(expr: pd.DataFrame, variance_quantile: float = 0.10) -> pd.DataFrame:
    """Remove genes below the variance_quantile threshold."""
    var = expr.var(axis=1)
    threshold = var.quantile(variance_quantile)
    keep = var >= threshold
    filtered = expr.loc[keep]
    print(f"Low-variance filter (q={variance_quantile}): "
          f"{expr.shape[0]} → {filtered.shape[0]} genes retained")
    return filtered


def remove_duplicate_genes(expr: pd.DataFrame, keep: str = 'highest_mean') -> pd.DataFrame:
    """Collapse duplicate gene symbols by keeping the row with the highest mean."""
    if not expr.index.duplicated().any():
        return expr
    if keep == 'highest_mean':
        expr = expr.loc[~expr.index.duplicated(keep='first')]   # fallback
        expr['_mean'] = expr.mean(axis=1)
        expr = expr.sort_values('_mean', ascending=False)
        expr = expr[~expr.index.duplicated(keep='first')]
        expr = expr.drop(columns='_mean')
    print(f"After deduplication: {expr.shape[0]} unique genes")
    return expr


def detect_outlier_samples(expr: pd.DataFrame, z_threshold: float = 3.0) -> list:
    """
    Detect outlier samples using correlation distance to the median profile.
    Returns a list of outlier sample names.
    """
    median_profile = expr.median(axis=1)
    correlations   = expr.apply(lambda col: col.corr(median_profile), axis=0)
    z_scores       = np.abs(stats.zscore(correlations))
    outliers       = correlations.index[z_scores > z_threshold].tolist()
    print(f"Outlier samples detected (|z|>{z_threshold}): {outliers}")
    return outliers


# ─────────────────────────────────────────────
# 3. NORMALISATION
# ─────────────────────────────────────────────

def log2_transform(expr: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """log2(x + pseudocount) transformation."""
    transformed = np.log2(expr + pseudocount)
    print(f"log2 transform applied (pseudocount={pseudocount})")
    return transformed


def quantile_normalize(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile normalisation across samples (columns).
    After normalisation every sample has the same distribution.
    """
    rank_mean = expr.stack().groupby(expr.rank(method='first').stack().astype(int)).mean()
    normalized = expr.rank(method='min').stack().astype(int).map(rank_mean).unstack()
    print("Quantile normalisation applied")
    return normalized


def zscore_normalize(expr: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """
    Z-score normalise.
    axis=0 → per-gene (across samples)
    axis=1 → per-sample (across genes)
    """
    if axis == 0:
        scaler = StandardScaler()
        normed = pd.DataFrame(scaler.fit_transform(expr.T).T,
                              index=expr.index, columns=expr.columns)
    else:
        normed = expr.apply(stats.zscore, axis=1, result_type='broadcast')
    print(f"Z-score normalisation applied (axis={axis})")
    return normed


def tpm_normalize(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
    """
    TPM normalisation for RNA-seq raw counts.
    gene_lengths : Series indexed by gene, lengths in base pairs.
    """
    common = counts.index.intersection(gene_lengths.index)
    counts = counts.loc[common]
    lengths = gene_lengths.loc[common]

    rpk   = counts.div(lengths / 1e3, axis=0)
    scale = rpk.sum(axis=0) / 1e6
    tpm   = rpk.div(scale, axis=1)
    print(f"TPM normalisation applied to {tpm.shape[0]} genes")
    return tpm


def robust_scale(expr: pd.DataFrame) -> pd.DataFrame:
    """RobustScaler (median / IQR) — less sensitive to outliers than z-score."""
    scaler = RobustScaler()
    scaled = pd.DataFrame(scaler.fit_transform(expr.T).T,
                          index=expr.index, columns=expr.columns)
    print("Robust scaling applied")
    return scaled


# ─────────────────────────────────────────────
# 4. BATCH CORRECTION (simple mean-centring)
# ─────────────────────────────────────────────

def mean_center_batches(expr: pd.DataFrame, batch_labels: pd.Series) -> pd.DataFrame:
    """
    Simple batch correction: subtract each batch's mean expression per gene.
    batch_labels : Series indexed by sample name.
    """
    corrected = expr.copy()
    for batch in batch_labels.unique():
        samples = batch_labels[batch_labels == batch].index
        batch_mean = expr[samples].mean(axis=1)
        corrected[samples] = expr[samples].sub(batch_mean, axis=0)
    print(f"Mean-centred {batch_labels.nunique()} batches")
    return corrected


# ─────────────────────────────────────────────
# 5. CONVENIENCE PIPELINE
# ─────────────────────────────────────────────

def full_preprocessing_pipeline(expr: pd.DataFrame,
                                  log_transform: bool = True,
                                  remove_low_expr: bool = True,
                                  remove_low_var: bool = True,
                                  normalise: str = 'zscore') -> pd.DataFrame:
    """
    One-shot preprocessing:
      1. (Optional) log2 transform
      2. (Optional) filter low-expression genes
      3. (Optional) filter low-variance genes
      4. Normalise (zscore | quantile | robust)

    Returns processed DataFrame.
    """
    print("\n── Full Preprocessing Pipeline ──")
    if log_transform:
        expr = log2_transform(expr)
    if remove_low_expr:
        expr = filter_low_expression(expr)
    if remove_low_var:
        expr = filter_low_variance(expr)

    if normalise == 'zscore':
        expr = zscore_normalize(expr)
    elif normalise == 'quantile':
        expr = quantile_normalize(expr)
    elif normalise == 'robust':
        expr = robust_scale(expr)

    print(f"Final matrix: {expr.shape[0]} genes × {expr.shape[1]} samples\n")
    return expr