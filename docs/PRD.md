# Hogan — Synthetic Data Generator

**Project:** Hogan (GAN-based tabular data synthesiser)
**Author:** Bobby DeVeaux
**Date:** 2026-04-29
**Status:** Draft / POC

---

## Problem

Real client data ends up in demos, dev environments, and git repos because generating realistic synthetic alternatives is hard. Teams rename files to "synthetic" but the data stays real. Compliance risk grows every time a CSV with real holdings, client names, or account IDs gets committed, shared on Slack, or loaded into a hackathon project.

**Concrete example:** `reinsight-ai` ships a `synthetic_holdings_mart.csv` with 10,693 rows of ClearWater-format holdings data containing real client names, account IDs, CUSIPs, market values, and credit ratings — pushed to a private GitHub repo.

## Solution

A CLI tool that learns the statistical shape of a real dataset and generates new rows that are structurally identical but contain no real data. Two commands:

```bash
hogan train holdings.csv              # Learn distributions, correlations, column types
hogan synthesise -n 10000 -o out.csv  # Generate 10,000 synthetic rows
```

The output CSV has the same columns, same value distributions, same inter-column correlations (e.g. duration vs. rating, country vs. currency), but every row is fabricated. No real client names, no real CUSIPs, no real account IDs.

## Goals

1. **Privacy by default** — output must not contain any verbatim real values for sensitive columns (names, IDs, accounts)
2. **Statistical fidelity** — distributions, correlations, and edge cases preserved well enough that downstream code (dashboards, analytics, ML models) works identically
3. **Zero config for common cases** — auto-detect column types (categorical, numeric, datetime, identifier) without manual schema definition
4. **Simple CLI** — two commands, minimal flags, fast iteration

## Non-Goals (POC)

- Real-time / streaming synthesis
- Multi-table relational synthesis (joins, foreign keys) — future feature
- Differential privacy guarantees — POC focuses on practical utility, not formal privacy proofs
- GPU acceleration — CPU-only for POC, GPU as optional optimisation later
- Web UI

## Architecture

### Technology

**Python** — the ML ecosystem for tabular GANs lives in Python (CTGAN, SDV, PyTorch). Go would be fighting upstream. The CLI wrapper can be thin.

### Core Components

```
hogan/
  cli/              # Click-based CLI entry point
    __init__.py
    main.py         # train, synthesise, inspect commands
  core/
    profiler.py     # Auto-detect column types, build metadata
    trainer.py      # CTGAN training wrapper
    synthesiser.py  # Generation + post-processing
    sanitiser.py    # Replace leaked real values, validate privacy
  models/           # Saved model artifacts (.pkl)
  config/
    defaults.yaml   # Default hyperparameters
```

### Data Flow

```
                    ┌─────────────┐
  real.csv ──────►  │  Profiler   │  Auto-detect column types
                    └──────┬──────┘
                           │ metadata.json
                    ┌──────▼──────┐
  hogan train ───►  │  Trainer    │  CTGAN fit (epochs, batch size)
                    └──────┬──────┘
                           │ model.pkl
                    ┌──────▼──────┐
  hogan synth ───►  │ Synthesiser │  Generate N rows
                    └──────┬──────┘
                           │ raw synthetic
                    ┌──────▼──────┐
                    │  Sanitiser  │  Validate no real values leaked
                    └──────┬──────┘
                           │
                      synthetic.csv
```

### Column Type Detection (Profiler)

| Type | Detection heuristic | Synthesis strategy |
|------|--------------------|--------------------|
| **Numeric continuous** | Float, high cardinality | GAN learns distribution |
| **Numeric discrete** | Int, low-medium cardinality | GAN learns distribution, round to int |
| **Categorical** | String, cardinality < 5% of rows | GAN learns frequency distribution |
| **Identifier** | String, cardinality ~ row count (unique per row) | Generate new UUIDs / sequential IDs — never copy real |
| **Name/Label** | String, moderate cardinality, heuristic (contains "client", "account", "name") | Faker-generated replacements, mapped consistently |
| **Date** | ISO date / datetime parseable | GAN learns temporal distribution |
| **Rating** | Matches known patterns (AAA, AA+, Baa1, etc.) | GAN learns ordinal distribution |

### Sanitiser

Post-generation safety net:

1. **Identifier columns** — assert zero overlap with training set
2. **Name columns** — assert zero overlap with training set
3. **Numeric columns** — warn if any exact row-vector matches the training set
4. **Report** — output a privacy summary: overlap %, column-by-column stats

## CLI Design

### `hogan train`

```
hogan train <input.csv> [options]

Options:
  --epochs INT        Training epochs (default: 300)
  --batch-size INT    Batch size (default: 500)
  --model-dir PATH    Where to save model (default: .hogan/)
  --name TEXT         Model name (default: derived from filename)
  --sensitive COLS    Comma-separated columns to force-treat as sensitive
  --id-cols COLS      Comma-separated columns to force-treat as identifiers
  --preview           Profile only — show detected types, don't train
```

Example:
```bash
hogan train holdings.csv --preview
# Shows: 46 columns detected
#   cw_client          → name (12 unique values)
#   cw_cusip           → identifier (8,921 unique)
#   cw_duration        → numeric_continuous (range: 0.01 - 32.4)
#   cw_s_and_p_rating  → categorical (12 categories: AAA, AA+, ...)
#   ...

hogan train holdings.csv --epochs 500 --sensitive cw_client,cw_account
# Training... [=========>         ] 62% (310/500 epochs, loss: 0.342)
# Model saved to .hogan/holdings/model.pkl
# Metadata saved to .hogan/holdings/metadata.json
```

### `hogan synthesise`

```
hogan synthesise [options]

Options:
  --model PATH        Model directory (default: .hogan/<most-recent>/)
  -n INT              Number of rows to generate (default: same as training set)
  -o PATH             Output file (default: stdout)
  --format TEXT       csv|parquet|json (default: csv)
  --seed INT          Random seed for reproducibility
  --validate          Run sanitiser and print privacy report
```

Example:
```bash
hogan synthesise -n 10000 -o synthetic_holdings.csv --validate
# Generating 10,000 rows...
# Privacy report:
#   cw_client:    0/10000 overlap with training (✓ all Faker-generated)
#   cw_cusip:     0/10000 overlap with training (✓ all generated)
#   cw_duration:  KS-test p=0.87 (✓ distribution preserved)
#   Row-vector:   0 exact matches (✓)
# Written to synthetic_holdings.csv
```

### `hogan inspect`

```
hogan inspect <synthetic.csv> --against <real.csv>

# Comparison report:
#   Column distributions: 44/46 within KS threshold
#   Correlation matrix RMSE: 0.032
#   Sensitive column overlap: 0%
#   Recommended: PASS
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python | CTGAN, SDV, PyTorch ecosystem. Go CLI would just shell out to Python anyway |
| GAN library | CTGAN (sdv-dev/CTGAN) | Purpose-built for tabular data, handles mixed types, well-maintained |
| Name generation | Faker | Industry standard for fake PII, locale-aware |
| CLI framework | Click | Clean, composable, good for POC |
| Model storage | pickle + JSON metadata | Simple, portable. Upgrade to ONNX later if needed |
| Packaging | pip install / pipx | Single `hogan` command after install |

## Success Criteria (POC)

1. `hogan train synthetic_holdings_mart.csv` completes in < 5 minutes on CPU
2. `hogan synthesise -n 10000` produces a CSV that:
   - Has identical column names and types
   - Passes schema validation against the original
   - Contains zero real client names or account IDs
   - Has similar distributions (KS-test p > 0.5 for numeric columns)
3. The reinsight-ai dashboard loads the synthetic CSV and works identically

## Future (Post-POC)

- **Multi-table** — define foreign key relationships, synthesise joined datasets
- **Conditional generation** — "generate 1000 rows where country=FR and rating<BBB"
- **Privacy budget** — formal differential privacy (DP-CTGAN)
- **GPU support** — CUDA acceleration for large datasets
- **MCP integration** — expose as a Cerebra tool ("synthesise this dataset")
- **Column constraints** — user-defined rules (e.g. "maturity_date > as_of_date")
- **Incremental training** — fine-tune on new data without full retrain

## References

- [CTGAN paper](https://arxiv.org/abs/1907.00503) — Modeling Tabular data using Conditional GAN
- [SDV (Synthetic Data Vault)](https://sdv.dev/) — broader framework, CTGAN is one engine
- [Faker](https://faker.readthedocs.io/) — fake data generation for PII columns
- Reinsight-AI holdings CSV — the motivating dataset (10,693 ClearWater-format rows)
