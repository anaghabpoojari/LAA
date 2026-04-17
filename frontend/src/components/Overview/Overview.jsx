import './Overview.css'

const cards = [
  {
    icon: '🧮',
    title: 'The Data',
    body: 'Expression matrix X ∈ ℝ^(m×n) — 57,736 gene probes × 283 cancer/healthy patients. Each entry X[i,j] = how active gene i is in patient j.',
    color: 'blue',
  },
  {
    icon: '🎯',
    title: 'The Goal',
    body: 'Identify cancer-discriminating genes purely from matrix mathematics — no biological prior knowledge. Let the numbers speak.',
    color: 'purple',
  },
  {
    icon: '💡',
    title: 'The Insight',
    body: 'Diseases occupy a low-dimensional subspace of the 15,000-dim gene space. SVD finds this subspace. Only 10–50 directions capture 90% of biology.',
    color: 'cyan',
  },
  {
    icon: '🧬',
    title: 'The Dataset',
    body: 'GSE68086 — real Tumor-Educated Platelet RNA-seq data from NCBI GEO (~34 MB). Blood-test detectable cancer signatures.',
    color: 'green',
  },
]

const techStack = [
  { lib: 'NumPy / SciPy', role: 'SVD, eigendecomposition, core LA' },
  { lib: 'scikit-learn', role: 'PCA, NMF, ICA, classifiers, CV' },
  { lib: 'pandas', role: 'Gene × sample DataFrame management' },
  { lib: 'matplotlib / seaborn', role: 'Heatmaps, scree, volcano plots' },
  { lib: 'networkx', role: 'Gene correlation network graphs' },
  { lib: 'statsmodels', role: 'FDR / Benjamini-Hochberg correction' },
  { lib: 'umap-learn', role: 'Non-linear manifold projection' },
  { lib: 'GEOparse', role: 'NCBI GEO dataset loading' },
]

export default function Overview() {
  return (
    <section id="overview" className="section overview-section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">THE PROBLEM</span>
          <h2>Why Linear Algebra for Biology?</h2>
          <p>Gene expression is a massive number table. Standard statistics fail at 15,000 dimensions. LA finds hidden disease structure.</p>
        </div>

        <div className="overview-cards">
          {cards.map(c => (
            <div key={c.title} className={`ov-card ov-${c.color}`}>
              <div className="ov-icon">{c.icon}</div>
              <h3>{c.title}</h3>
              <p>{c.body}</p>
            </div>
          ))}
        </div>

        {/* SVD equation visual */}
        <div className="svd-eq-block">
          <div className="svd-eq-label">Core Decomposition</div>
          <div className="svd-eq">
            <div className="svd-matrix tall red-m">
              <span className="m-name">X</span>
              <span className="m-dim">m × n</span>
            </div>
            <span className="eq-sign">=</span>
            <div className="svd-matrix tall blue-m">
              <span className="m-name">U</span>
              <span className="m-dim">m × k</span>
            </div>
            <span className="eq-sign">·</span>
            <div className="svd-matrix sq purple-m">
              <span className="m-name">Σ</span>
              <span className="m-dim">k × k</span>
            </div>
            <span className="eq-sign">·</span>
            <div className="svd-matrix wide green-m">
              <span className="m-name">Vᵀ</span>
              <span className="m-dim">k × n</span>
            </div>
          </div>
          <div className="svd-labels">
            <span>Gene space</span>
            <span>Singular values</span>
            <span>Sample space</span>
          </div>
        </div>

        {/* Tech stack */}
        <div className="tech-stack">
          <div className="ts-title">Technology Stack</div>
          <div className="ts-grid">
            {techStack.map(t => (
              <div key={t.lib} className="ts-item">
                <code className="ts-lib">{t.lib}</code>
                <span className="ts-role">{t.role}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
