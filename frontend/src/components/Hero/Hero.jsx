import { useEffect, useRef, useState } from 'react'
import './Hero.css'

const STATS = [
  { target: 57736, label: 'Genes (Probes)', suffix: '' },
  { target: 283, label: 'Patient Samples', suffix: '' },
  { target: 7, label: 'Notebooks', suffix: '' },
  { target: 4, label: 'LA Techniques', suffix: '' },
]

function useCounter(target, started) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (!started) return
    let start = null
    const duration = 1800
    const step = (ts) => {
      if (!start) start = ts
      const prog = Math.min((ts - start) / duration, 1)
      const ease = 1 - Math.pow(1 - prog, 3)
      setVal(Math.round(ease * target))
      if (prog < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [started, target])
  return val
}

function StatItem({ target, label, suffix, started }) {
  const val = useCounter(target, started)
  return (
    <div className="hero-stat">
      <span className="stat-num">{val.toLocaleString()}{suffix}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}

export default function Hero() {
  const canvasRef = useRef(null)
  const [started, setStarted] = useState(false)

  // Matrix rain canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let animId

    const resize = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const chars = 'ATCGUΣΛΨΩΠΣΔΦ01ABCDEFXYZsvdpcaλσμβρ'
    const fontSize = 13
    let cols = Math.floor(canvas.width / fontSize)
    let drops = Array(cols).fill(1).map(() => Math.random() * -100)

    const draw = () => {
      ctx.fillStyle = 'rgba(3,3,13,0.04)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      cols = Math.floor(canvas.width / fontSize)
      if (drops.length < cols) drops = [...drops, ...Array(cols - drops.length).fill(1)]

      for (let i = 0; i < cols; i++) {
        const char = chars[Math.floor(Math.random() * chars.length)]
        const alpha = Math.random() * 0.4 + 0.05
        ctx.fillStyle = i % 5 === 0
          ? `rgba(0,212,255,${alpha})`
          : `rgba(124,111,255,${alpha})`
        ctx.font = `${fontSize}px 'JetBrains Mono', monospace`
        ctx.fillText(char, i * fontSize, drops[i] * fontSize)
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0
        drops[i]++
      }
      animId = requestAnimationFrame(draw)
    }
    draw()

    setTimeout(() => setStarted(true), 300)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <section id="hero" className="hero">
      <canvas ref={canvasRef} className="hero-canvas" />

      <div className="hero-overlay" />

      <div className="hero-content container">
        <div className="hero-badge">
          <span className="badge-dot" />
          Semester 4 · Linear Algebra Project
        </div>

        <h1 className="hero-title">
          <span>Gene Expression</span>
          <span>Disease Discovery</span>
          <span>via <span className="gradient-text">Linear Algebra</span></span>
        </h1>

        <p className="hero-sub">
          Decomposing a <strong>57,736 × 283</strong> patient matrix using{' '}
          <code>SVD</code>, <code>PCA</code>, <code>NMF</code> and{' '}
          <code>Eigenvalue Analysis</code> to reveal cancer-linked gene patterns
          invisible to the naked eye.
        </p>

        <div className="hero-formula-strip">
          <span className="formula-pill">X = UΣVᵀ</span>
          <span className="formula-sep">·</span>
          <span className="formula-pill">C = QΛQᵀ</span>
          <span className="formula-sep">·</span>
          <span className="formula-pill">X ≈ WH</span>
          <span className="formula-sep">·</span>
          <span className="formula-pill">λ_max = σ²(1+√γ)²</span>
        </div>

        <div className="hero-stats">
          {STATS.map(s => (
            <StatItem key={s.label} {...s} started={started} />
          ))}
        </div>

        <div className="hero-actions">
          <a href="#howtorun" className="btn-primary">🚀 Run Locally</a>
          <a href="#math" className="btn-secondary">📐 Explore Math</a>
          <a
            href="https://github.com/anaghabpoojari/LAA"
            target="_blank"
            rel="noreferrer"
            className="btn-ghost"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
        </div>
      </div>

      <div className="scroll-hint">scroll to explore ↓</div>
    </section>
  )
}
