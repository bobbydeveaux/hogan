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

/* ── Benchmark bar component ── */
function BenchBar({ label, python, rust, unit }: { label: string; python: number; rust: number; unit: string }) {
  const speedup = python / rust
  const maxVal = Math.max(python, rust)
  return (
    <div className="bench-row">
      <div className="bench-label">{label}</div>
      <div className="bench-bars">
        <div className="bench-bar-wrap">
          <div className="bench-bar python" style={{ width: `${(python / maxVal) * 100}%` }}>
            <span>{python}{unit}</span>
          </div>
          <span className="bench-lang">Python</span>
        </div>
        <div className="bench-bar-wrap">
          <div className="bench-bar rust" style={{ width: `${(rust / maxVal) * 100}%` }}>
            <span>{rust}{unit}</span>
          </div>
          <span className="bench-lang">Rust</span>
        </div>
      </div>
      <div className="bench-speedup">{speedup.toFixed(0)}x</div>
    </div>
  )
}

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
              <a href="#v2">v2 Rust</a>
              <a href="#benchmarks">Benchmarks</a>
              <a href="#transform">How It Works</a>
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
            <span className="hero-badge">v2 &mdash; Now in Rust</span>
            <h1>
              <span className="gradient">41x faster.</span><br />
              Zero compromises.
            </h1>
            <p className="hero-sub">
              The world&rsquo;s first compiled CTGAN. Hogan v2 rewrites the entire GAN pipeline in Rust,
              generating 18,594 synthetic rows per second. Single binary. No Python. No runtime.
            </p>
            <div className="hero-actions">
              <a href="#install" className="btn-primary">Get Started</a>
              <a href="#benchmarks" className="btn-secondary">See Benchmarks</a>
            </div>

            {/* Terminal demo */}
            <div className="terminal">
              <div className="terminal-bar">
                <span className="terminal-dot red" />
                <span className="terminal-dot amber" />
                <span className="terminal-dot green" />
              </div>
              <div className="terminal-body">
                <div><span className="prompt">$</span> <span className="cmd">hogan train holdings.csv --epochs 150</span></div>
                <div><span className="comment"># 10,693 rows, 50 columns | Transform: 0.26s | GMM fit: 0.81s</span></div>
                <div><span className="success">Training completed in 463s</span></div>
                <div><span className="success">Model saved to .hogan-rs/model.msgpack</span></div>
                <br />
                <div><span className="prompt">$</span> <span className="cmd">hogan synthesise -n 10000 -o synthetic.csv</span></div>
                <div><span className="success">Generated 10,000 rows in 0.538s</span> <span className="value">(18,594 rows/sec)</span></div>
                <br />
                <div><span className="comment"># Python equivalent: 22.4s</span></div>
                <div><span className="warn">Rust is 16x faster end-to-end, 41x faster at generation</span></div>
              </div>
            </div>
          </div>
        </section>

        {/* ── v2 Announcement ── */}
        <section className="morph-section" id="v2">
          <div className="section">
            <div className="section-label">Hogan v2</div>
            <div className="section-title">They said GANs can&rsquo;t be done in Rust.</div>
            <div className="section-desc">
              We built the first compiled CTGAN implementation from scratch &mdash; Conditional Tabular GAN
              with mode-specific normalisation, WGAN-GP training, PacGAN discriminator, and Gumbel-Softmax
              activation. All in Rust. All from first principles.
            </div>

            <div className="v2-highlights">
              <div className="v2-card">
                <div className="v2-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                </div>
                <h3>Pure Compiled Binary</h3>
                <p>Single executable. No Python interpreter, no pip, no venv, no dependency hell. Copy the binary and run.</p>
              </div>
              <div className="v2-card">
                <div className="v2-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                </div>
                <h3>18,594 Rows/sec</h3>
                <p>Synthesis is 41x faster than Python. Data transform is 8x faster. Training is 1.4x faster. Every operation is compiled and optimised.</p>
              </div>
              <div className="v2-card">
                <div className="v2-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                </div>
                <h3>Full CTGAN from Scratch</h3>
                <p>Not a wrapper. We ported the entire CTGAN architecture: EM-based GMM, residual generator, PacGAN discriminator, WGAN-GP gradient penalty.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ── Benchmarks ── */}
        <section className="section" id="benchmarks">
          <div className="section-label">Benchmarks</div>
          <div className="section-title">The numbers speak for themselves.</div>
          <div className="section-desc">
            Tested on 10,693 real ClearWater-format holdings rows, 50 columns.
            150 training epochs on Apple M-series CPU.
          </div>

          <div className="bench-section">
            <BenchBar label="Synthesis (10k rows)" python={22.4} rust={1.4} unit="s" />
            <BenchBar label="Generation only" python={15} rust={0.54} unit="s" />
            <BenchBar label="Data transform" python={2} rust={0.26} unit="s" />
            <BenchBar label="GMM fitting" python={3} rust={0.81} unit="s" />
          </div>

          <div className="stats-bar" style={{ marginTop: '3rem' }}>
            <div className="stat">
              <div className="stat-value">41x</div>
              <div className="stat-label">Faster Generation</div>
            </div>
            <div className="stat">
              <div className="stat-value">18,594</div>
              <div className="stat-label">Rows/sec</div>
            </div>
            <div className="stat">
              <div className="stat-value">0%</div>
              <div className="stat-label">Real Data Leaked</div>
            </div>
            <div className="stat">
              <div className="stat-value">1</div>
              <div className="stat-label">Binary. That&rsquo;s It.</div>
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

              <div className="transform-arrow">
                <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M8 24h28M28 16l8 8-8 8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <circle cx="24" cy="24" r="22" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
                </svg>
                <span>CTGAN</span>
              </div>

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

        {/* ── How it works ── */}
        <section className="section">
          <div className="section-label">Architecture</div>
          <div className="section-title">CTGAN. From first principles. In Rust.</div>
          <div className="section-desc">
            Not a Python binding. A ground-up implementation of the Conditional Tabular GAN paper.
          </div>

          <div className="steps">
            <div className="step">
              <div className="step-number">1</div>
              <h3>EM-Based GMM</h3>
              <p>Mode-specific normalisation via a custom Gaussian Mixture Model. Detects multimodal distributions in continuous columns and normalises within each mode.</p>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <h3>Residual Generator</h3>
              <p>Concatenation-based residual blocks with BatchNorm and Gumbel-Softmax activation. Dimensions grow through the network for richer representations.</p>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <h3>PacGAN Discriminator</h3>
              <p>Groups of 10 samples judged together to prevent mode collapse. WGAN-GP gradient penalty enforces the Lipschitz constraint without weight clipping.</p>
            </div>
            <div className="step">
              <div className="step-number">4</div>
              <h3>Training-by-Sampling</h3>
              <p>Conditional vectors force the generator to practice every category. Log-frequency sampling ensures minority classes are well-represented.</p>
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
              <p>Identifiers get fresh values, names get Faker replacements, and the sanitiser validates zero overlap with training data. No real values leak.</p>
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
              <h3>Dual Runtime</h3>
              <p>Choose Python (pip install) for prototyping or the compiled Rust binary for production. Same CLI interface, same results, dramatically different speed.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon purple">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
              </div>
              <h3>Compiled Performance</h3>
              <p>The Rust binary generates 18,594 rows/sec&mdash;41x faster than Python. Data preprocessing is 8x faster. No interpreter overhead, no GIL.</p>
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
          <div className="section-title">Two runtimes. Your choice.</div>

          <div className="install-grid">
            <div className="code-block">
              <div className="code-header">
                <span>Rust (Recommended)</span>
                <span className="tag tag-synth" style={{ fontSize: '0.6rem' }}>41x Faster</span>
              </div>
              <div className="code-body">
                <div><span className="prompt">$</span> <span className="cmd">git clone https://github.com/bobbydeveaux/hogan.git</span></div>
                <div><span className="prompt">$</span> <span className="cmd">cd hogan/hogan-rs</span></div>
                <div><span className="prompt">$</span> <span className="cmd">cargo build --release</span></div>
                <br />
                <div><span className="comment"># Train</span></div>
                <div><span className="prompt">$</span> <span className="cmd">./target/release/hogan train data.csv</span></div>
                <br />
                <div><span className="comment"># Synthesise</span></div>
                <div><span className="prompt">$</span> <span className="cmd">./target/release/hogan synthesise -n 10000 -o out.csv</span></div>
              </div>
            </div>

            <div className="code-block">
              <div className="code-header">
                <span>Python</span>
              </div>
              <div className="code-body">
                <div><span className="prompt">$</span> <span className="cmd">pip install hogan</span></div>
                <br />
                <div><span className="comment"># Train</span></div>
                <div><span className="prompt">$</span> <span className="cmd">hogan train data.csv --epochs 300</span></div>
                <br />
                <div><span className="comment"># Synthesise</span></div>
                <div><span className="prompt">$</span> <span className="cmd">hogan synthesise -n 10000 -o out.csv</span></div>
                <br />
                <div><span className="comment"># Compare</span></div>
                <div><span className="prompt">$</span> <span className="cmd">hogan inspect out.csv --against data.csv</span></div>
              </div>
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="cta-section">
          <h2>Stop shipping real data.</h2>
          <p>
            Every CSV with real client names in a dev environment is a compliance incident waiting to happen.
            Hogan makes it trivially easy to generate safe alternatives &mdash; at compiled speed.
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
            <span className="footer-copy">Hogan &mdash; Synthetic data, real privacy. Compiled speed.</span>
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
