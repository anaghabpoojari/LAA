"""
validation.py
Statistical validation functions for gene expression / LA analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, fisher_exact
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DIFFERENTIAL EXPRESSION
# ─────────────────────────────────────────────

def t_test_de(expr: pd.DataFrame,
              group_labels: pd.Series,
              group_a: str, group_b: str,
              fdr_method: str = 'fdr_bh') -> pd.DataFrame:
    """
    Welch t-test for differential expression between two groups.
    Returns a DataFrame sorted by adjusted p-value.
    """
    a_samples = group_labels[group_labels == group_a].index
    b_samples = group_labels[group_labels == group_b].index

    a_expr = expr[a_samples]
    b_expr = expr[b_samples]

    results = []
    for gene in expr.index:
        t, p = stats.ttest_ind(a_expr.loc[gene], b_expr.loc[gene],
                                equal_var=False, nan_policy='omit')
        log2fc = np.log2((a_expr.loc[gene].mean() + 1e-9) /
                          (b_expr.loc[gene].mean() + 1e-9))
        results.append({'gene': gene, 'log2FoldChange': log2fc,
                        'pvalue': p, 'tstat': t})

    df = pd.DataFrame(results).set_index('gene')
    _, df['padj'], _, _ = multipletests(df['pvalue'].fillna(1), method=fdr_method)
    df = df.sort_values('padj')
    print(f"DE ({group_a} vs {group_b}): "
          f"{(df['padj'] < 0.05).sum()} significant genes (FDR < 0.05)")
    return df


def mann_whitney_de(expr: pd.DataFrame,
                    group_labels: pd.Series,
                    group_a: str, group_b: str,
                    fdr_method: str = 'fdr_bh') -> pd.DataFrame:
    """Non-parametric Mann–Whitney U test for differential expression."""
    a_samples = group_labels[group_labels == group_a].index
    b_samples = group_labels[group_labels == group_b].index

    results = []
    for gene in expr.index:
        a_vals = expr.loc[gene, a_samples].dropna()
        b_vals = expr.loc[gene, b_samples].dropna()
        try:
            u, p = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        except Exception:
            u, p = np.nan, 1.0
        fc = np.log2((a_vals.mean() + 1e-9) / (b_vals.mean() + 1e-9))
        results.append({'gene': gene, 'log2FoldChange': fc, 'pvalue': p, 'U': u})

    df = pd.DataFrame(results).set_index('gene')
    _, df['padj'], _, _ = multipletests(df['pvalue'].fillna(1), method=fdr_method)
    df = df.sort_values('padj')
    print(f"Mann-Whitney DE: {(df['padj'] < 0.05).sum()} significant genes")
    return df


# ─────────────────────────────────────────────
# 2. COMPONENT ASSOCIATION TESTS
# ─────────────────────────────────────────────

def pc_group_anova(scores: pd.DataFrame,
                   group_labels: pd.Series) -> pd.DataFrame:
    """
    One-way ANOVA / Kruskal-Wallis for each PC against group labels.
    Returns p-values (raw + FDR) for each component.
    """
    rows = []
    common = scores.index.intersection(group_labels.index)
    scores = scores.loc[common]
    labels = group_labels.loc[common]

    for pc in scores.columns:
        groups = [scores.loc[labels == g, pc].values
                  for g in labels.unique()]
        _, p_anova = stats.f_oneway(*groups)
        _, p_kw    = kruskal(*groups)
        rows.append({'component': pc, 'p_anova': p_anova, 'p_kruskal': p_kw})

    df = pd.DataFrame(rows).set_index('component')
    _, df['padj_anova'], _, _   = multipletests(df['p_anova'], method='fdr_bh')
    _, df['padj_kruskal'], _, _ = multipletests(df['p_kruskal'], method='fdr_bh')
    sig = (df['padj_anova'] < 0.05).sum()
    print(f"ANOVA: {sig}/{len(df)} components significantly associated with groups")
    return df.sort_values('p_anova')


def pc_continuous_correlation(scores: pd.DataFrame,
                               continuous_var: pd.Series,
                               method: str = 'pearson') -> pd.DataFrame:
    """Correlate each PC with a continuous clinical variable."""
    common = scores.index.intersection(continuous_var.index)
    scores = scores.loc[common]
    var    = continuous_var.loc[common]
    rows   = []

    for pc in scores.columns:
        if method == 'pearson':
            r, p = stats.pearsonr(scores[pc], var)
        else:
            r, p = stats.spearmanr(scores[pc], var)
        rows.append({'component': pc, 'r': r, 'pvalue': p})

    df = pd.DataFrame(rows).set_index('component')
    _, df['padj'], _, _ = multipletests(df['pvalue'], method='fdr_bh')
    return df.sort_values('pvalue')


# ─────────────────────────────────────────────
# 3. CLASSIFIER VALIDATION
# ─────────────────────────────────────────────

def cross_validate_classifier(clf, X: np.ndarray, y: np.ndarray,
                                n_splits: int = 5,
                                scoring: str = 'roc_auc') -> dict:
    """
    Stratified k-fold cross-validation.
    Returns mean ± std of the chosen metric and per-fold scores.
    """
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    result = {
        'mean':   scores.mean(),
        'std':    scores.std(),
        'scores': scores,
        'metric': scoring,
    }
    print(f"CV ({n_splits}-fold) {scoring}: "
          f"{scores.mean():.4f} ± {scores.std():.4f}")
    return result


def evaluate_binary_classifier(clf, X_train, y_train,
                                 X_test, y_test) -> dict:
    """Fit and evaluate a binary classifier. Returns metrics dict."""
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    return {
        'auc': auc, 'confusion_matrix': cm,
        'report': report, 'fpr': fpr, 'tpr': tpr,
        'thresholds': thresholds,
    }


# ─────────────────────────────────────────────
# 4. GENE SET ENRICHMENT (simple overlap)
# ─────────────────────────────────────────────

def hypergeometric_enrichment(gene_list: list,
                               gene_sets: dict,
                               background_size: int) -> pd.DataFrame:
    """
    Simple hypergeometric / Fisher-exact enrichment for a list of genes
    against a dict of gene sets {name: [gene1, gene2, ...]}.
    """
    rows = []
    N = background_size
    K = len(gene_list)
    gene_set_genes = set(gene_list)

    for name, gs in gene_sets.items():
        gs   = set(gs)
        M    = len(gs)
        k    = len(gs & gene_set_genes)
        # Fisher's exact
        tbl  = [[k, M - k], [K - k, N - M - K + k]]
        _, p = fisher_exact(tbl, alternative='greater')
        rows.append({
            'gene_set': name,
            'overlap':  k,
            'set_size': M,
            'pvalue':   p,
        })

    df = pd.DataFrame(rows).set_index('gene_set')
    _, df['padj'], _, _ = multipletests(df['pvalue'], method='fdr_bh')
    return df.sort_values('padj')


# ─────────────────────────────────────────────
# 5. BOOTSTRAP STABILITY
# ─────────────────────────────────────────────

def bootstrap_pca_stability(expr: pd.DataFrame,
                              n_bootstraps: int = 100,
                              n_components: int = 10) -> pd.DataFrame:
    """
    Bootstrap stability of PCA loadings.
    Returns average cosine similarity of each PC loading across bootstrap runs.
    """
    from sklearn.decomposition import PCA
    from src.linear_algebra import cosine_similarity_matrix

    ref_pca = PCA(n_components=n_components, random_state=42)
    X       = expr.T.values.astype(float)
    ref_pca.fit(X)
    ref_load = ref_pca.components_      # k × genes

    sims = np.zeros((n_bootstraps, n_components))
    rng  = np.random.default_rng(0)

    for b in range(n_bootstraps):
        idx  = rng.integers(0, X.shape[0], size=X.shape[0])
        Xb   = X[idx]
        pca  = PCA(n_components=n_components, random_state=b)
        pca.fit(Xb)
        for k in range(n_components):
            cos = abs(np.dot(ref_load[k], pca.components_[k]) /
                      (np.linalg.norm(ref_load[k]) *
                       np.linalg.norm(pca.components_[k]) + 1e-12))
            sims[b, k] = cos

    stability = pd.DataFrame({
        f'PC{i+1}': sims[:, i] for i in range(n_components)
    })
    print("Bootstrap PCA stability (mean cosine similarity):")
    print(stability.mean().round(4).to_string())
    return stability


# ─────────────────────────────────────────────
# 6. PERMUTATION TEST FOR COMPONENT ASSOCIATION
# ─────────────────────────────────────────────

def permutation_test_auc(clf, X: np.ndarray, y: np.ndarray,
                          n_permutations: int = 1000,
                          n_splits: int = 5) -> dict:
    """
    Permutation test: compare observed cross-val AUC to null distribution.
    Returns observed AUC, null AUCs, and empirical p-value.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    obs  = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean()
    null = np.zeros(n_permutations)
    rng  = np.random.default_rng(1)

    for i in range(n_permutations):
        yp   = rng.permutation(y)
        null[i] = cross_val_score(clf, X, yp, cv=cv, scoring='roc_auc').mean()

    p_value = (null >= obs).mean()
    print(f"Observed AUC: {obs:.4f} | Permutation p-value: {p_value:.4f}")
    return {'observed_auc': obs, 'null_aucs': null, 'p_value': p_value}