"""
linear_algebra.py
Linear-algebra utility functions for gene expression analysis.
Covers: SVD, PCA, NMF, ICA, CCA, eigenvalue decomposition,
        gene loading extraction, and dimensionality-reduction helpers.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA, NMF, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. SVD  (full & truncated)
# ─────────────────────────────────────────────

def run_svd(expr: pd.DataFrame, n_components: int = 50) -> dict:
    """
    Truncated SVD on the expression matrix (genes × samples).
    Returns U (gene space), S (singular values), Vt (sample space).
    """
    X = expr.values.astype(float)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)

    U  = svd.components_.T                        # genes  × k
    S  = svd.singular_values_                      # k
    Vt = svd.transform(X) / S                     # samples × k  (right singular vectors)

    explained = svd.explained_variance_ratio_

    print(f"SVD: top-{n_components} components explain "
          f"{explained.sum()*100:.1f}% variance")

    return {
        'U':        U,          # gene   loadings
        'S':        S,          # singular values
        'Vt':       Vt,         # sample scores
        'explained_variance_ratio': explained,
        'gene_names':   expr.index.tolist(),
        'sample_names': expr.columns.tolist(),
    }


def svd_explained_variance_table(svd_result: dict) -> pd.DataFrame:
    """Return a DataFrame of cumulative explained variance per component."""
    evr = svd_result['explained_variance_ratio']
    df  = pd.DataFrame({
        'component':          np.arange(1, len(evr)+1),
        'explained_variance': evr,
        'cumulative':         np.cumsum(evr),
    })
    return df


def select_n_components_by_variance(svd_result: dict,
                                     threshold: float = 0.90) -> int:
    """Return the number of SVD components needed to explain ≥ threshold variance."""
    cum = np.cumsum(svd_result['explained_variance_ratio'])
    n   = int(np.searchsorted(cum, threshold)) + 1
    print(f"Components to explain ≥{threshold*100:.0f}% variance: {n}")
    return n


# ─────────────────────────────────────────────
# 2. PCA
# ─────────────────────────────────────────────

def run_pca(expr: pd.DataFrame, n_components: int = 50,
            center: bool = True, scale: bool = False) -> dict:
    """
    PCA on the expression matrix (genes × samples).
    By default centres (not scales) the data before decomposition.
    Returns sample scores, gene loadings, and explained variance.
    """
    X = expr.T.values.astype(float)   # samples × genes

    if center or scale:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        X = scaler.fit_transform(X)

    pca   = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)    # samples × k

    result = {
        'scores':    pd.DataFrame(scores,
                                  index=expr.columns,
                                  columns=[f'PC{i+1}' for i in range(n_components)]),
        'loadings':  pd.DataFrame(pca.components_.T,
                                  index=expr.index,
                                  columns=[f'PC{i+1}' for i in range(n_components)]),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'explained_variance':       pca.explained_variance_,
        'pca_object': pca,
    }

    print(f"PCA: top-{n_components} PCs explain "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance")
    return result


def pca_gene_contributions(pca_result: dict, pc: int = 1, top_n: int = 20) -> pd.DataFrame:
    """Return top_n genes contributing most to a given principal component (1-indexed)."""
    col  = f'PC{pc}'
    load = pca_result['loadings'][col].abs().sort_values(ascending=False)
    return pca_result['loadings'].loc[load.index[:top_n], [col]]


def project_new_samples(pca_result: dict, new_expr: pd.DataFrame) -> pd.DataFrame:
    """Project new samples (genes × samples) into the PCA space."""
    pca = pca_result['pca_object']
    X   = new_expr.T.values.astype(float)
    proj = pca.transform(X)
    return pd.DataFrame(proj, index=new_expr.columns,
                        columns=pca_result['scores'].columns)


# ─────────────────────────────────────────────
# 3. NMF  (metagenes)
# ─────────────────────────────────────────────

def run_nmf(expr: pd.DataFrame, n_components: int = 10,
            max_iter: int = 500, init: str = 'nndsvda') -> dict:
    """
    Non-negative matrix factorisation: X ≈ W · H
      W  (genes × k)  — metagene signatures
      H  (k × samples) — sample metagene usage
    Requires non-negative input — shifts data if needed.
    """
    X = expr.values.astype(float)
    if X.min() < 0:
        X = X - X.min()          # shift to non-negative

    nmf = NMF(n_components=n_components, init=init,
              max_iter=max_iter, random_state=42)
    W   = nmf.fit_transform(X)   # genes × k
    H   = nmf.components_        # k × samples

    recon_error = nmf.reconstruction_err_
    print(f"NMF ({n_components} components): reconstruction error = {recon_error:.4f}")

    return {
        'W': pd.DataFrame(W, index=expr.index,
                          columns=[f'Metagene{i+1}' for i in range(n_components)]),
        'H': pd.DataFrame(H,
                          index=[f'Metagene{i+1}' for i in range(n_components)],
                          columns=expr.columns),
        'reconstruction_error': recon_error,
        'nmf_object': nmf,
    }


def top_metagene_genes(nmf_result: dict, metagene: int = 1,
                        top_n: int = 30) -> pd.Series:
    """Return top_n genes with highest weight for a given metagene (1-indexed)."""
    col = f'Metagene{metagene}'
    return nmf_result['W'][col].sort_values(ascending=False).head(top_n)


# ─────────────────────────────────────────────
# 4. ICA  (independent components)
# ─────────────────────────────────────────────

def run_ica(expr: pd.DataFrame, n_components: int = 20,
            max_iter: int = 1000) -> dict:
    """
    FastICA on the expression matrix.
    Returns mixing matrix (genes × k) and sample activations (samples × k).
    """
    X   = expr.T.values.astype(float)   # samples × genes
    ica = FastICA(n_components=n_components, max_iter=max_iter, random_state=42)
    S   = ica.fit_transform(X)          # samples × k  (source signals)
    A   = ica.mixing_                   # genes × k   (mixing / loading matrix)

    print(f"ICA: {n_components} independent components extracted")
    return {
        'sources':  pd.DataFrame(S, index=expr.columns,
                                 columns=[f'IC{i+1}' for i in range(n_components)]),
        'mixing':   pd.DataFrame(A, index=expr.index,
                                 columns=[f'IC{i+1}' for i in range(n_components)]),
        'ica_object': ica,
    }


# ─────────────────────────────────────────────
# 5. CORRELATION & EIGENVALUE ANALYSIS
# ─────────────────────────────────────────────

def gene_correlation_matrix(expr: pd.DataFrame,
                              method: str = 'pearson') -> pd.DataFrame:
    """Compute gene–gene correlation matrix (genes × genes)."""
    if method == 'pearson':
        corr = expr.T.corr()
    else:
        corr = expr.T.corr(method='spearman')
    print(f"{method.capitalize()} correlation matrix: {corr.shape}")
    return corr


def sample_correlation_matrix(expr: pd.DataFrame,
                               method: str = 'pearson') -> pd.DataFrame:
    """Compute sample–sample correlation matrix (samples × samples)."""
    corr = expr.corr(method=method)
    print(f"Sample {method} correlation matrix: {corr.shape}")
    return corr


def eigenvalue_decomposition(corr_matrix: pd.DataFrame) -> dict:
    """
    Eigenvalue decomposition of a (symmetric) correlation matrix.
    Returns eigenvalues (descending) and eigenvectors.
    """
    vals, vecs = linalg.eigh(corr_matrix.values)
    idx   = np.argsort(vals)[::-1]
    vals  = vals[idx]
    vecs  = vecs[:, idx]

    total  = vals.sum()
    evr    = vals / total

    return {
        'eigenvalues':              vals,
        'eigenvectors':             vecs,
        'explained_variance_ratio': evr,
        'cumulative_variance':      np.cumsum(evr),
        'labels':                   corr_matrix.index.tolist(),
    }


def marchenko_pastur_threshold(n_genes: int, n_samples: int,
                                sigma: float = 1.0) -> float:
    """
    Marchenko–Pastur upper bound — the maximum eigenvalue expected under a
    random (null) correlation model. Eigenvalues above this are 'signal'.
    """
    ratio = n_genes / n_samples
    lmax  = sigma**2 * (1 + np.sqrt(ratio))**2
    print(f"Marchenko–Pastur threshold: λ_max = {lmax:.4f}")
    return lmax


def significant_eigengenes(eig_result: dict, threshold: float) -> np.ndarray:
    """Return indices of eigenvalues that exceed a given threshold."""
    idx = np.where(eig_result['eigenvalues'] > threshold)[0]
    print(f"Significant eigengenes (λ > {threshold:.3f}): {len(idx)}")
    return idx


# ─────────────────────────────────────────────
# 6. GENE LOADING EXTRACTION
# ─────────────────────────────────────────────

def get_top_genes_by_loading(loadings: pd.DataFrame, component: str,
                              top_n: int = 50, absolute: bool = True) -> pd.DataFrame:
    """
    Extract top_n genes by their loading magnitude on a given component.
    loadings : DataFrame (genes × components)
    component: column name, e.g. 'PC1', 'Metagene3'
    """
    col  = loadings[component]
    col  = col.abs() if absolute else col
    top  = col.sort_values(ascending=False).head(top_n)
    return loadings.loc[top.index, [component]]


def biplot_loadings(pca_result: dict, pc_x: int = 1, pc_y: int = 2,
                    top_n: int = 15) -> pd.DataFrame:
    """
    Return gene loadings on two PCs for biplot visualisation.
    Scaled to unit circle.
    """
    cx, cy = f'PC{pc_x}', f'PC{pc_y}'
    load   = pca_result['loadings'][[cx, cy]].copy()
    # scale so max loading = 1
    for c in [cx, cy]:
        load[c] = load[c] / load[c].abs().max()
    # select genes with largest combined magnitude
    load['mag'] = np.sqrt(load[cx]**2 + load[cy]**2)
    return load.nlargest(top_n, 'mag').drop(columns='mag')


# ─────────────────────────────────────────────
# 7. MATRIX UTILITIES
# ─────────────────────────────────────────────

def condition_number(matrix: np.ndarray) -> float:
    """Return the condition number κ = σ_max / σ_min."""
    s  = np.linalg.svd(matrix, compute_uv=False)
    kn = s.max() / (s.min() + 1e-12)
    print(f"Condition number: {kn:.2f}")
    return kn


def effective_rank(singular_values: np.ndarray, threshold: float = 0.99) -> int:
    """Number of singular values needed to capture ≥ threshold of total energy."""
    energy = np.cumsum(singular_values**2) / (singular_values**2).sum()
    return int(np.searchsorted(energy, threshold)) + 1


def reconstruct_from_components(U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                                  k: int) -> np.ndarray:
    """Low-rank reconstruction using the top-k SVD components."""
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def cosine_similarity_matrix(A: np.ndarray) -> np.ndarray:
    """Column-wise cosine similarity matrix."""
    norms = np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
    An    = A / norms
    return An.T @ An