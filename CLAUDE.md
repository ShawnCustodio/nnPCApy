# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnPCApy is a Python port of the R `nsprcomp` algorithm (non-negative sparse PCA) with an application layer for gene set scoring in epithelial-mesenchymal transition (EMT) research.

Three sections:
1. **Core library** (`src/nnpcapy/`): installable package — nsprcomp EM solver + three scoring methods
2. **Application** (`emtscore/`): full EMTscore workflow (bulk + single-cell analysis, plotting)
3. **Benchmark** (`paper/benchmark/`): R vs Python head-to-head timing comparison (moved here from
   the former top-level `benchmark/` during the `paper/`-reorg)

## Commands

```bash
# Install (editable, with dev + benchmark extras)
pip install -e ".[benchmark,dev]"

# Run tests
pytest tests/

# Run a single test
pytest tests/test_basic.py::test_pc1_nonneg

# Lint
ruff check src/ emtscore/ tests/

# Run benchmark (in order)
python paper/benchmark/datasets.py      # cache .npy inputs (~5 s)
python paper/benchmark/bench_python.py  # time Python (~3 min)
Rscript paper/benchmark/bench_r.R       # time R (~8 min, requires nsprcomp + RcppCNPy)
python paper/benchmark/compare.py       # merge CSVs and generate plots
```

## Architecture

### Core algorithm: `src/nnpcapy/nsprcomp.py`

`nsprcomp(x, ncomp, center, scale_, nneg, nrestart=5, em_tol=1e-3, em_maxiter=100)` — EM solver with Gram-Schmidt orthogonalization, optional non-negativity constraint on loadings, multiple random restarts, and deflation-based multi-component extraction. Returns `{"x": scores, "rotation": loadings, "sdev", "center", "scale"}`.

Three changes take our naive first port to the shipping code (cumulative median 6.8× per call over the naive port; total-wall 19× faster than R). **Honest framing — only #3 is a genuine algorithmic change vs R.** #1 and #2 bring our port up to parity with R's *existing* defaults and algorithm; they are not gains over R:

1. **Parameter alignment with R** (`nrestart` 20→5, `em_tol` 1e-4→1e-3, `em_maxiter` 200→100): these are R `nsprcomp`'s own defaults — our initial naive port used more conservative values. Reproducing R, not improving on it.

2. **Two-pass support refinement / variational renormalization** (`_empca_cov_refined`): each restart runs EM once on the full d-dimensional problem to find the non-zero support (nneg zeroes ~half of features), then re-runs EM on the support submatrix for refined weights. This is exactly what R `nsprcomp` does (Moghaddam 2006) — reproduced, not novel.

3. **Covariance precomputation** (`C = Xp.T @ Xp` once per component) — the one algorithmic change vs R: iterate `w ∝ C w` (O(d²)/iter, one GEMV over the d×d covariance) instead of two GEMV over the data matrix (O(n×d)/iter). Break-even is ≈ d/2 iterations, amortized across all restarts and iterations of a component (~5 × 100–200 reuses of the one C). Per-iteration cost falls by ~n/d (≈60× at n=12k, d=200); end-to-end this is ~5× at single-cell scale (Fig 2A). The two-pass submatrix is then a free `C[np.ix_(supp, supp)]` slice rather than a re-scan of Xp. The covariance form is noted by Sigg & Buhmann but not used by the R package.

So the speedup vs R decomposes as **vectorization (R→V2, ~4×) × covariance (V2→V3, ~5×)** at the single-cell ablation point (N=12k, D=200; total ≈21×, robust to gene-set size — the mix shifts with D but the product does not); the naive→shipping journey (params + two-pass) is our port converging to R, not a gain over R. Benchmark numbers (total-wall 19×; single gene sets median ~19×, multi ~8×, single-cell ~12×, bulk ~1.6×; memory ~4–78× less heap) all trace to `paper/benchmark/results/summary.csv`.

### Gene set scoring methods (parallel implementations in both `src/nnpcapy/` and `emtscore/`)

- **nnPCA** (`nnpca.py`): calls nsprcomp on gene subsets; `run_nnPCA()` scores each gene set independently then collapses with PCA
- **AUCell** (`aucell.py`): fraction of signature genes in top-expressed genes (default 5% threshold)
- **ssGSEA** (`ssGSEA.py`): weighted Kolmogorov-Smirnov running enrichment score

### Dual-module pattern

`src/nnpcapy/` is the pip-installable library. `emtscore/` is a parallel application layer with the same modules extended with workflow logic (e.g. `emtscore/nsprcomp.py` adds `compute_M1_M2_scores()`). The `emtscore/workflow.py` facade re-exports all functions; `emtscore/pipeline.py` orchestrates the numbered analysis sections (2.1–3.6).

### Benchmark

`paper/benchmark/datasets.py` caches 47 normalized matrices as `.npy` files. Both `bench_python.py` and `bench_r.R` read identical inputs and run 705 calls each (47 matrices × 3 ncomps × 5 trials). `compare.py` merges the result CSVs and writes plots to `paper/benchmark/plots/`. The Python-only speedup ablation behind Fig 2A (V0→V3, no R required) lives in `paper/benchmark/synth/`.

### Data

Bundled in `data/`: bulk expression (`geneExp.csv`, 120 cell lines × ~16k genes), E/M signatures (`Panchy_et_al_*.csv`), GMT files (`EM_signature.gmt`, `filtered.c2.gmt`), and Cook 2020 single-cell datasets (`cook2020/*.csv`). `EM_signature.gmt` is generated at runtime by `pipeline.load_bulk_data()`.

## Figure output

Save all analysis figures in both `.png` and `.svg` formats.
