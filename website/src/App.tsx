import { useEffect, useRef } from 'react'

/* ── Matrix rain canvas ── */
function MatrixRain() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const chars = '01ABCDEF{}[]<>:.;|/\\+-*=%$#@!~'
    const fontSize = 14
    const columns = Math.floor(canvas.width / fontSize)
    const drops: number[] = Array(columns).fill(1).map(() => Math.random() * -100)

    const draw = () => {
      ctx.fillStyle = 'rgba(6, 6, 10, 0.05)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      for (let i = 0; i < drops.length; i++) {
        const char = chars[Math.floor(Math.random() * chars.length)]
        const x = i * fontSize
        const y = drops[i] * fontSize

        // Gradient: purple at top, cyan at bottom
        const ratio = (y / canvas.height)
        const r = Math.floor(168 - ratio * 162)
        const g = Math.floor(85 + ratio * 97)
        const b = Math.floor(247 - ratio * 35)
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.4 + Math.random() * 0.3})`
        ctx.font = `${fontSize}px JetBrains Mono, monospace`
        ctx.fillText(char, x, y)

        if (y > canvas.height && Math.random() > 0.975) {
          drops[i] = 0
        }
        drops[i] += 0.5 + Math.random() * 0.5
      }
    }

    const interval = setInterval(draw, 50)
    return () => { clearInterval(interval); window.removeEventListener('resize', resize) }
  }, [])

  return <canvas ref={canvasRef} id="matrix-canvas" />
}

/* ── Data for the demo panels ── */
const realRows = [
  ['John Smith', '521598', '12189LBL4', '$666,932'],
  ['InEvo Re Ltd', '445676', 'F858D3RT2', '$8,131,490'],
  ['RAIF Macquarie', '445686', '04018YAP3', '$3,282,028'],
  ['Jane Wilson', '389012', 'G5218QRT1', '$12,450,100'],
]

const synthRows = [
  ['Pennington LLC', '738201', 'CU7GE8BKI', '$671,204'],
  ['Grant-Floyd Inc', '294857', '0KMVKIQ4F', '$8,290,112'],
  ['Hughes-Best Co', '619384', 'DDVHX48TT', '$3,198,445'],
  ['Quinn-Martinez', '502716', 'MEBJQLVL4', '$12,881,330'],
]

function App() {
  return (
    <>
      <MatrixRain />
      <div className="page-content">

        {/* ── Nav ── */}
        <nav>
          <div className="nav-inner">
            <span className="nav-logo">hogan</span>
            <div className="nav-links">
              <a href="#transform">How It Works</a>
              <a href="#features">Features</a>
              <a href="#install">Install</a>
              <a href="https://github.com/bobbydeveaux/hogan" className="btn-github">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
                GitHub
              </a>
            </div>
          </div>
        </nav>

        {/* ── Hero ── */}
        <section className="hero">
          <div className="hero-content">
            <span className="hero-badge">GAN-Powered Synthesis</span>
            <h1>
              Your data.<br />
              <span className="gradient">None of the risk.</span>
            </h1>
            <p className="hero-sub">
              Train on real datasets. Generate statistically identical synthetic rows.
              Zero real names, zero real IDs, zero compliance nightmares.
            </p>
            <div className="hero-actions">
              <a href="#install" className="btn-primary">Get Started</a>
              <a href="https://github.com/bobbydeveaux/hogan" className="btn-secondary">View on GitHub</a>
            </div>

            {/* Terminal demo */}
            <div className="terminal">
              <div className="terminal-bar">
                <span className="terminal-dot red" />
                <span className="terminal-dot amber" />
                <span className="terminal-dot green" />
              </div>
              <div className="terminal-body">
                <div><span className="prompt">$</span> <span className="cmd">hogan train holdings.csv --epochs 300</span></div>
                <div><span className="comment"># Training CTGAN on 10,693 rows, 50 columns</span></div>
                <div><span className="comment"># Discrete columns: 26 | Epochs: 300</span></div>
                <div><span className="success">Model saved to .hogan/holdings/</span></div>
                <br />
                <div><span className="prompt">$</span> <span className="cmd">hogan synthesise -n 10000 -o synthetic.csv --validate</span></div>
                <div><span className="success">Generated 10,000 rows</span></div>
                <div><span className="comment"># Privacy report:</span></div>
                <div>&nbsp; cw_client: &nbsp;&nbsp;&nbsp;<span className="success">0/10000 overlap</span> <span className="comment">(all Faker-generated)</span></div>
                <div>&nbsp; cw_cusip: &nbsp;&nbsp;&nbsp;&nbsp;<span className="success">0/10000 overlap</span> <span className="comment">(all generated)</span></div>
                <div>&nbsp; cw_duration: &nbsp;<span className="value">KS-test p=0.87</span> <span className="comment">(distribution preserved)</span></div>
                <div>&nbsp; Row-vector: &nbsp;&nbsp;<span className="success">0 exact matches</span></div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Transform visualisation ── */}
        <section className="morph-section" id="transform">
          <div className="section">
            <div className="section-label">The Transformation</div>
            <div className="section-title">Real data in. Synthetic data out.</div>
            <div className="section-desc">
              Hogan learns the statistical shape of your dataset&mdash;distributions, correlations,
              edge cases&mdash;then generates entirely new rows. No real values survive.
            </div>

            <div className="data-table-demo">
              {/* Real data panel */}
              <div className="data-panel">
                <div className="data-panel-header">
                  <span className="label">Real Data</span>
                  <span className="tag tag-real">Sensitive</span>
                </div>
                <div className="data-rows">
                  <div className="data-row header">
                    <span>Client</span><span>Account</span><span>CUSIP</span><span>Value</span>
                  </div>
                  {realRows.map((row, i) => (
                    <div className="data-row" key={i}>
                      <span className="sensitive">{row[0]}</span>
                      <span className="sensitive">{row[1]}</span>
                      <span className="sensitive">{row[2]}</span>
                      <span>{row[3]}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Arrow */}
              <div className="transform-arrow">
                <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M8 24h28M28 16l8 8-8 8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <circle cx="24" cy="24" r="22" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
                </svg>
                <span>CTGAN</span>
              </div>

              {/* Synthetic data panel */}
              <div className="data-panel">
                <div className="data-panel-header">
                  <span className="label">Synthetic Data</span>
                  <span className="tag tag-synth">Safe</span>
                </div>
                <div className="data-rows">
                  <div className="data-row header">
                    <span>Client</span><span>Account</span><span>CUSIP</span><span>Value</span>
                  </div>
                  {synthRows.map((row, i) => (
                    <div className="data-row" key={i}>
                      <span className="safe">{row[0]}</span>
                      <span className="safe">{row[1]}</span>
                      <span className="safe">{row[2]}</span>
                      <span>{row[3]}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Stats ── */}
        <div className="section">
          <div className="stats-bar">
            <div className="stat">
              <div className="stat-value">0%</div>
              <div className="stat-label">Real Data Leaked</div>
            </div>
            <div className="stat">
              <div className="stat-value">50+</div>
              <div className="stat-label">Column Types Detected</div>
            </div>
            <div className="stat">
              <div className="stat-value">2</div>
              <div className="stat-label">Commands to Ship</div>
            </div>
            <div className="stat">
              <div className="stat-value">&lt;5m</div>
              <div className="stat-label">Training on CPU</div>
            </div>
          </div>
        </div>

        {/* ── How it works ── */}
        <section className="section">
          <div className="section-label">How It Works</div>
          <div className="section-title">Four stages. Full pipeline.</div>
          <div className="section-desc">
            From raw CSV to privacy-safe synthetic data in minutes.
          </div>

          <div className="steps">
            <div className="step">
              <div className="step-number">1</div>
              <h3>Profile</h3>
              <p>Auto-detect column types: numeric, categorical, identifier, name, date, credit rating. Zero config for common cases.</p>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <h3>Train</h3>
              <p>CTGAN learns the joint distribution of your data&mdash;correlations between columns, multimodal distributions, category frequencies.</p>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <h3>Synthesise</h3>
              <p>Generate any number of new rows. Identifiers get fresh UUIDs, names get Faker replacements, numerics come from the learned distribution.</p>
            </div>
            <div className="step">
              <div className="step-number">4</div>
              <h3>Sanitise</h3>
              <p>Post-generation safety net validates zero overlap on sensitive columns, KS-tests on numerics, and checks for row-vector matches.</p>
            </div>
          </div>
        </section>

        {/* ── Features ── */}
        <section className="section" id="features">
          <div className="section-label">Features</div>
          <div className="section-title">Built for real-world data privacy.</div>
          <div className="section-desc">
            Not a toy. A tool that handles the messy reality of production datasets.
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon purple">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
              </div>
              <h3>Smart Column Profiling</h3>
              <p>Automatically detects numeric, categorical, identifier, name, date, and credit rating columns. Handles mixed types without manual schemas.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon cyan">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              </div>
              <h3>Privacy by Default</h3>
              <p>Identifiers get fresh UUIDs, names get Faker replacements, and the sanitiser validates zero overlap with training data. No real values leak.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon green">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/></svg>
              </div>
              <h3>Statistical Fidelity</h3>
              <p>CTGAN preserves distributions, correlations, and edge cases. KS-tests verify numeric columns match. Downstream dashboards work identically.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon magenta">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>
              </div>
              <h3>Zero Config CLI</h3>
              <p>Two commands: <code>train</code> and <code>synthesise</code>. Preview mode shows detected types before training. Sensible defaults for everything.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon purple">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              </div>
              <h3>CPU-First Performance</h3>
              <p>Trains on 10,000+ rows in under 5 minutes on a laptop CPU. No GPU required for the POC. CUDA support planned for large datasets.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon cyan">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              </div>
              <h3>Flexible Output</h3>
              <p>Export to CSV, Parquet, or JSON. Set row counts, random seeds for reproducibility, and run the sanitiser with a single flag.</p>
            </div>
          </div>
        </section>

        {/* ── Install ── */}
        <section className="section" id="install">
          <div className="section-label">Get Started</div>
          <div className="section-title">Up and running in 30 seconds.</div>

          <div className="code-block">
            <div className="code-header">
              <span>Install</span>
            </div>
            <div className="code-body">
              <div><span className="prompt">$</span> <span className="cmd">pip install hogan</span></div>
              <br />
              <div><span className="comment"># Or from source</span></div>
              <div><span className="prompt">$</span> <span className="cmd">git clone https://github.com/bobbydeveaux/hogan.git</span></div>
              <div><span className="prompt">$</span> <span className="cmd">cd hogan && pip install -e .</span></div>
            </div>
          </div>

          <div className="code-block" style={{ marginTop: '1.5rem' }}>
            <div className="code-header">
              <span>Usage</span>
            </div>
            <div className="code-body">
              <div><span className="comment"># Preview column detection</span></div>
              <div><span className="prompt">$</span> <span className="cmd">hogan train data.csv --preview</span></div>
              <br />
              <div><span className="comment"># Train the model</span></div>
              <div><span className="prompt">$</span> <span className="cmd">hogan train data.csv --epochs 300</span></div>
              <br />
              <div><span className="comment"># Generate synthetic data</span></div>
              <div><span className="prompt">$</span> <span className="cmd">hogan synthesise -n 10000 -o synthetic.csv</span></div>
              <br />
              <div><span className="comment"># Compare against original</span></div>
              <div><span className="prompt">$</span> <span className="cmd">hogan inspect synthetic.csv --against data.csv</span></div>
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="cta-section">
          <h2>Stop shipping real data.</h2>
          <p>
            Every CSV with real client names in a dev environment is a compliance incident waiting to happen.
            Hogan makes it trivially easy to generate safe alternatives.
          </p>
          <div className="hero-actions">
            <a href="https://github.com/bobbydeveaux/hogan" className="btn-primary">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
              Star on GitHub
            </a>
            <a href="#install" className="btn-secondary">Get Started</a>
          </div>
        </section>

        {/* ── Footer ── */}
        <footer>
          <div className="footer-inner">
            <span className="footer-copy">Hogan &mdash; Synthetic data, real privacy.</span>
            <div className="footer-links">
              <a href="https://github.com/bobbydeveaux/hogan">GitHub</a>
              <a href="https://github.com/bobbydeveaux/guardian">Guardian</a>
              <a href="https://github.com/bobbydeveaux/pulse">Pulse</a>
            </div>
          </div>
        </footer>

      </div>
    </>
  )
}

export default App
