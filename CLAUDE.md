# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnPCApy is a Python port of the R `nsprcomp` algorithm (non-negative sparse PCA) with an application layer for gene set scoring in epithelial-mesenchymal transition (EMT) research.

Three sections:
1. **Core library** (`src/nnpcapy/`): installable package — nsprcomp EM solver + three scoring methods
2. **Application** (`emtscore/`): full EMTscore workflow (bulk + single-cell analysis, plotting)
3. **Benchmark** (`benchmark/`): R vs Python head-to-head timing comparison

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
python benchmark/datasets.py      # cache .npy inputs (~5 s)
python benchmark/bench_python.py  # time Python (~3 min)
Rscript benchmark/bench_r.R       # time R (~8 min, requires nsprcomp + RcppCNPy)
python benchmark/compare.py       # merge CSVs and generate plots
```

## Architecture

### Core algorithm: `src/nnpcapy/nsprcomp.py`

`nsprcomp(x, ncomp, center, scale_, nneg, nrestart=5, em_tol=1e-3, em_maxiter=100)` — EM solver with Gram-Schmidt orthogonalization, optional non-negativity constraint on loadings, multiple random restarts, and deflation-based multi-component extraction. Returns `{"x": scores, "rotation": loadings, "sdev", "center", "scale"}`.

Three layered optimizations (applied in order, cumulative 6.8× speedup over the naive port, 19× faster than R):

1. **Parameter alignment with R** (`nrestart` 20→5, `em_tol` 1e-4→1e-3, `em_maxiter` 200→100): eliminates 4× redundant restarts that dominated cost on small matrices.

2. **Two-pass support refinement** (`_empca_cov_refined`): each restart runs EM once on the full d-dimensional problem to find the non-zero support (nneg zeroes ~half of features), then re-runs EM on the smaller support submatrix for refined weights. Matches the strategy used by the R `nsprcomp` package.

3. **Covariance precomputation** (`C = Xp.T @ Xp` once per component): converts each EM iteration from O(n×d) (two GEMV over the data matrix) to O(d²) (one GEMV over the d×d covariance). Break-even is ~6–12 iterations; we do ~100–200. For single-cell matrices (n=12k, d=89) this reduces per-iteration cost by 177×. The two-pass second-pass submatrix is then a free `C[np.ix_(supp, supp)]` slice rather than a re-scan of Xp.

### Gene set scoring methods (parallel implementations in both `src/nnpcapy/` and `emtscore/`)

- **nnPCA** (`nnpca.py`): calls nsprcomp on gene subsets; `run_nnPCA()` scores each gene set independently then collapses with PCA
- **AUCell** (`aucell.py`): fraction of signature genes in top-expressed genes (default 5% threshold)
- **ssGSEA** (`ssgsea.py`): weighted Kolmogorov-Smirnov running enrichment score

### Dual-module pattern

`src/nnpcapy/` is the pip-installable library. `emtscore/` is a parallel application layer with the same modules extended with workflow logic (e.g. `emtscore/nsprcomp.py` adds `compute_M1_M2_scores()`). The `emtscore/workflow.py` facade re-exports all functions; `emtscore/pipeline.py` orchestrates the numbered analysis sections (2.1–3.6).

### Benchmark

`benchmark/datasets.py` caches 47 normalized matrices as `.npy` files. Both `bench_python.py` and `bench_r.R` read identical inputs and run 705 calls each (47 matrices × 3 ncomps × 5 trials). `compare.py` merges the result CSVs and writes plots to `benchmark/plots/`.

### Data

Bundled in `data/`: bulk expression (`geneExp.csv`, 120 cell lines × ~16k genes), E/M signatures (`Panchy_et_al_*.csv`), GMT files (`EM_signature.gmt`, `filtered.c2.gmt`), and Cook 2020 single-cell datasets (`cook2020/*.csv`). `EM_signature.gmt` is generated at runtime by `pipeline.load_bulk_data()`.

## Figure output

Save all analysis figures in both `.png` and `.svg` formats.
