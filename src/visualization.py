"""
visualization.py
Plotting utilities for gene expression linear-algebra analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = sns.color_palette('tab10')

# ─────────────────────────────────────────────
# 1. SCREE / VARIANCE EXPLAINED
# ─────────────────────────────────────────────

def plot_scree(explained_variance_ratio: np.ndarray,
               title: str = 'Scree Plot',
               max_components: int = 30,
               save_path: str = None):
    """Bar + cumulative line scree plot."""
    evr = explained_variance_ratio[:max_components]
    cum = np.cumsum(evr)
    x   = np.arange(1, len(evr)+1)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x, evr * 100, color='steelblue', alpha=0.7, label='Individual')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Explained Variance (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, cum * 100, 'o-', color='crimson', label='Cumulative')
    ax2.axhline(90, ls='--', color='grey', lw=0.8, label='90%')
    ax2.set_ylabel('Cumulative Variance (%)', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, 105)

    plt.title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 2. PCA SCATTER PLOTS
# ─────────────────────────────────────────────

def plot_pca_scatter(scores: pd.DataFrame,
                     labels: pd.Series = None,
                     pc_x: int = 1, pc_y: int = 2,
                     title: str = 'PCA — Sample Scatter',
                     save_path: str = None):
    """2D PCA scatter with optional group colouring."""
    cx, cy = f'PC{pc_x}', f'PC{pc_y}'
    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        groups = labels.unique()
        palette = dict(zip(groups, PALETTE))
        for g in groups:
            idx = labels[labels == g].index
            s   = scores.loc[idx]
            ax.scatter(s[cx], s[cy], label=str(g), s=60, alpha=0.8,
                       color=palette[g])
        ax.legend(title='Group', bbox_to_anchor=(1.01, 1), loc='upper left')
    else:
        ax.scatter(scores[cx], scores[cy], s=60, alpha=0.7, color='steelblue')

    ax.set_xlabel(cx)
    ax.set_ylabel(cy)
    ax.set_title(title)
    ax.axhline(0, lw=0.5, color='grey')
    ax.axvline(0, lw=0.5, color='grey')
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_3d(scores: pd.DataFrame, labels: pd.Series = None,
                title: str = 'PCA 3D'):
    """Interactive-style 3D PCA scatter (matplotlib)."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    if labels is not None:
        groups  = labels.unique()
        palette = dict(zip(groups, PALETTE))
        for g in groups:
            idx = labels[labels == g].index
            s   = scores.loc[idx]
            ax.scatter(s['PC1'], s['PC2'], s['PC3'],
                       label=str(g), s=50, alpha=0.8, color=palette[g])
        ax.legend()
    else:
        ax.scatter(scores['PC1'], scores['PC2'], scores['PC3'], s=50, alpha=0.7)

    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 3. HEATMAPS
# ─────────────────────────────────────────────

def plot_expression_heatmap(expr: pd.DataFrame,
                             row_labels: pd.Series = None,
                             col_labels: pd.Series = None,
                             top_n_genes: int = 100,
                             title: str = 'Expression Heatmap',
                             save_path: str = None):
    """Clustered heatmap of top-variable genes."""
    var = expr.var(axis=1).sort_values(ascending=False)
    top = expr.loc[var.index[:top_n_genes]]

    row_colors = None
    if col_labels is not None:
        lut = dict(zip(col_labels.unique(),
                       sns.color_palette('Set2', col_labels.nunique())))
        row_colors = col_labels.map(lut)

    g = sns.clustermap(top, cmap='RdBu_r', center=0,
                        col_colors=row_colors,
                        xticklabels=False, yticklabels=False,
                        figsize=(12, 8),
                        dendrogram_ratio=0.1,
                        cbar_pos=(0.02, 0.8, 0.03, 0.18))
    g.fig.suptitle(title, y=1.01)
    if save_path: g.fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              title: str = 'Correlation Matrix',
                              save_path: str = None):
    """Heatmap of a (sample or gene) correlation matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask    = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.3,
                ax=ax, cbar_kws={'shrink': 0.7})
    ax.set_title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 4. NMF / METAGENE PLOTS
# ─────────────────────────────────────────────

def plot_metagene_weights(W: pd.DataFrame, metagene: int = 1,
                           top_n: int = 25,
                           save_path: str = None):
    """Horizontal bar chart of top gene weights for one metagene."""
    col   = f'Metagene{metagene}'
    genes = W[col].sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    bars = ax.barh(genes.index[::-1], genes.values[::-1],
                   color='teal', alpha=0.8)
    ax.set_xlabel('NMF Weight')
    ax.set_title(f'Top {top_n} Genes — {col}')
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_nmf_H_heatmap(H: pd.DataFrame, col_labels: pd.Series = None,
                        title: str = 'NMF Sample Usage (H matrix)',
                        save_path: str = None):
    """Heatmap of the NMF H matrix (metagenes × samples)."""
    fig, ax = plt.subplots(figsize=(max(10, H.shape[1] // 3), 5))
    sns.heatmap(H, cmap='YlOrRd', ax=ax,
                xticklabels=False, linewidths=0,
                cbar_kws={'label': 'Usage'})
    ax.set_title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 5. EIGENVALUE / SPECTRUM PLOTS
# ─────────────────────────────────────────────

def plot_eigenvalue_spectrum(eigenvalues: np.ndarray,
                              mp_threshold: float = None,
                              title: str = 'Eigenvalue Spectrum',
                              save_path: str = None):
    """Bar chart of eigenvalues with optional Marchenko-Pastur threshold."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(1, len(eigenvalues)+1), eigenvalues,
           color='slateblue', alpha=0.7)
    if mp_threshold is not None:
        ax.axhline(mp_threshold, ls='--', color='crimson',
                   label=f'M-P threshold ({mp_threshold:.2f})')
        ax.legend()
    ax.set_xlabel('Eigenvalue rank')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title)
    ax.set_yscale('log')
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 6. BIPLOT
# ─────────────────────────────────────────────

def plot_biplot(scores: pd.DataFrame, loadings: pd.DataFrame,
                labels: pd.Series = None,
                pc_x: int = 1, pc_y: int = 2,
                top_n_genes: int = 15,
                title: str = 'PCA Biplot',
                save_path: str = None):
    """Classic PCA biplot: sample scores + gene loading arrows."""
    cx, cy = f'PC{pc_x}', f'PC{pc_y}'
    fig, ax = plt.subplots(figsize=(9, 7))

    # samples
    if labels is not None:
        for g in labels.unique():
            idx = labels[labels == g].index
            ax.scatter(scores.loc[idx, cx], scores.loc[idx, cy],
                       label=str(g), s=50, alpha=0.7)
        ax.legend(title='Group')
    else:
        ax.scatter(scores[cx], scores[cy], s=50, alpha=0.7, color='steelblue')

    # gene arrows
    sc = scores[[cx, cy]].abs().values.max()
    ld = loadings[[cx, cy]].copy()
    ld['mag'] = np.sqrt(ld[cx]**2 + ld[cy]**2)
    top = ld.nlargest(top_n_genes, 'mag')

    arrow_scale = sc / top['mag'].max() * 0.8
    for gene, row in top.iterrows():
        ax.annotate('', xy=(row[cx]*arrow_scale, row[cy]*arrow_scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2))
        ax.text(row[cx]*arrow_scale*1.05, row[cy]*arrow_scale*1.05,
                gene, fontsize=7, color='crimson')

    ax.axhline(0, lw=0.5, color='grey')
    ax.axvline(0, lw=0.5, color='grey')
    ax.set_xlabel(cx); ax.set_ylabel(cy)
    ax.set_title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 7. GENE LOADING BAR CHART
# ─────────────────────────────────────────────

def plot_gene_loadings(loadings_col: pd.Series, top_n: int = 30,
                        title: str = 'Gene Loadings',
                        save_path: str = None):
    """Diverging bar chart of gene loadings for one component."""
    top = loadings_col.abs().sort_values(ascending=False).head(top_n)
    vals = loadings_col.loc[top.index]

    fig, ax = plt.subplots(figsize=(7, max(5, top_n * 0.28)))
    colors = ['crimson' if v > 0 else 'steelblue' for v in vals.values[::-1]]
    ax.barh(vals.index[::-1], vals.values[::-1], color=colors, alpha=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Loading')
    ax.set_title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 8. NETWORK / CORRELATION GRAPH
# ─────────────────────────────────────────────

def plot_correlation_network(corr_matrix: pd.DataFrame,
                              threshold: float = 0.7,
                              title: str = 'Gene Correlation Network',
                              save_path: str = None):
    """Draw a gene–gene correlation network (edges above threshold)."""
    try:
        import networkx as nx
    except ImportError:
        print("Install networkx: pip install networkx")
        return

    G = nx.Graph()
    genes = corr_matrix.index.tolist()
    G.add_nodes_from(genes)

    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            w = corr_matrix.iloc[i, j]
            if abs(w) >= threshold:
                G.add_edge(genes[i], genes[j], weight=w)

    if G.number_of_edges() == 0:
        print(f"No edges above threshold {threshold}. Lower the threshold.")
        return

    pos    = nx.spring_layout(G, seed=42, k=0.5)
    edges  = G.edges(data=True)
    colors = ['crimson' if d['weight'] > 0 else 'steelblue' for _, _, d in edges]
    widths = [abs(d['weight']) * 2 for _, _, d in edges]

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='gold',
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.set_title(f'{title}  (|r| ≥ {threshold})')
    ax.axis('off')
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ─────────────────────────────────────────────
# 9. VOLCANO PLOT
# ─────────────────────────────────────────────

def plot_volcano(de_df: pd.DataFrame,
                 log2fc_col: str = 'log2FoldChange',
                 pval_col: str   = 'padj',
                 fc_thresh: float = 1.0,
                 p_thresh: float  = 0.05,
                 label_top: int   = 15,
                 title: str = 'Volcano Plot',
                 save_path: str = None):
    """Volcano plot for differential expression results."""
    df = de_df.copy().dropna(subset=[log2fc_col, pval_col])
    df['neg_log10p'] = -np.log10(df[pval_col].clip(lower=1e-300))

    sig_up   = (df[log2fc_col] >  fc_thresh) & (df[pval_col] < p_thresh)
    sig_down = (df[log2fc_col] < -fc_thresh) & (df[pval_col] < p_thresh)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df.loc[~(sig_up | sig_down), log2fc_col],
               df.loc[~(sig_up | sig_down), 'neg_log10p'],
               s=15, alpha=0.4, color='grey', label='NS')
    ax.scatter(df.loc[sig_up, log2fc_col],
               df.loc[sig_up, 'neg_log10p'],
               s=20, alpha=0.7, color='crimson', label=f'Up ({sig_up.sum()})')
    ax.scatter(df.loc[sig_down, log2fc_col],
               df.loc[sig_down, 'neg_log10p'],
               s=20, alpha=0.7, color='steelblue', label=f'Down ({sig_down.sum()})')

    # label top genes
    top_genes = df.loc[sig_up | sig_down].nlargest(label_top, 'neg_log10p')
    for gene, row in top_genes.iterrows():
        ax.text(row[log2fc_col], row['neg_log10p'], str(gene),
                fontsize=6.5, ha='left', va='bottom')

    ax.axvline( fc_thresh, ls='--', lw=0.8, color='grey')
    ax.axvline(-fc_thresh, ls='--', lw=0.8, color='grey')
    ax.axhline(-np.log10(p_thresh), ls='--', lw=0.8, color='grey')
    ax.set_xlabel('log₂ Fold Change')
    ax.set_ylabel('-log₁₀(adjusted p-value)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 10. UMAP
# ─────────────────────────────────────────────

def plot_umap(embedding: np.ndarray, labels: pd.Series = None,
              title: str = 'UMAP Embedding',
              save_path: str = None):
    """2D UMAP scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        groups  = labels.unique()
        palette = dict(zip(groups, PALETTE))
        for g in groups:
            idx = np.where(labels == g)[0]
            ax.scatter(embedding[idx, 0], embedding[idx, 1],
                       s=40, alpha=0.8, label=str(g), color=palette[g])
        ax.legend(title='Group', bbox_to_anchor=(1.01, 1), loc='upper left')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=40, alpha=0.7)

    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()