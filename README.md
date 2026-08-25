This is a Python library with a companion manuscript under review.

nnPCApy: a Python library for fast non-negative principal component analysis and gene-set scoring

Shawn K Custodio, Zhao Sun, Tian Hong

# nnPCApy

A Python implementation of nonnegative, sparse PCA (non-negative sparse PCA) 
and the EMT scoring pipeline built on top of it, plus a benchmark comparing the
Python kernel against the original R implementation (nsprcomp by Sigg 2018).

This repo holds two pieces: the kernel (nnPCA) as an installable
library, and the full EMTscore-style application that uses it.

## Layout

```
src/nnpcapy/    # installable library: nsprcomp + nnPCA/AUCell/ssGSEA scoring
emtscore/       # EMT scoring application layer built on nnpcapy: workflow,
                # pipeline orchestration, plotting, GMM clustering
utility/        # data path resolution, Cook 2020 raw-data loaders
data/           # bundled signatures, GMT files, bulk + Cook SC data
notebooks/      # two demo notebooks that reproduce the EMTscore vignette
paper/          # public reproduction path for the manuscript:
                #   paper/benchmark/ - R-vs-Python timing harness (was
                #     top-level benchmark/ before the reorg)
                #   paper/panelE/    - Figure 3E showcase + Cook data packing
tests/          # tests for nsprcomp, nnPCA, AUCell, ssGSEA
REVIEW.md       # reviewer checklist for a clean-room install + sanity checks
```

## Install

The core solver is dependency-light (NumPy only):

```
pip install nnpcapy          # PyPI (once released), or:
pip install -e .             # from a clone -- just the nsprcomp solver
```

For the full toolkit (gene-set scoring, plotting, single-cell loaders):

```
pip install "nnpcapy[full]"          # PyPI (once released), or from a clone:
pip install -e ".[full,dev]"
```

The wheel ships the `emtscore` application layer alongside the `nnpcapy` solver,
so `pip install "nnpcapy[viz]"` alone is enough to score signatures and make the
Figure-3E-style plots on your own data — no checkout required.

Extras: `app` (pandas/scipy for scoring), `viz` (matplotlib/seaborn/scipy for
plots), `sc` (anndata/scikit-learn for single-cell), `full` = all three.

`from nnpcapy import nsprcomp` (core solver) and `import emtscore` (application +
plotting) both work. `import emtscore` is light -- the heavy dependencies load
only when a function that needs them is called.

### Bundled vs. checkout data

The wheel bundles the *small* data files (the E/M signatures, cell annotation,
and the `.gmt`/`.tsv` gene-set files), so signature-based scoring works straight
after install. The *large* files — bulk expression (`geneExp.csv`), the full
MSigDB C2 `.gmt`, and the Cook et al. 2020 single-cell matrices — are **not**
shipped; they come with a source checkout (under `data/`) or via a directory you
point `$NNPCAPY_DATA` at. Functions that need them raise a clear error explaining
where to get them.

## Use the library

```python
import numpy as np
from nnpcapy import nsprcomp

X = np.random.default_rng(0).standard_normal((100, 25))
out = nsprcomp(X, ncomp=2, nneg=True, center=True, scale_=False)

scores   = out["x"]         # (n_samples, ncomp)
loadings = out["rotation"]  # (n_features, ncomp), non-negative on PC1
```

Call signature mirrors R's `nsprcomp::nsprcomp(...)`.

## Score gene signatures and plot (single-cell)

The `emtscore` layer wraps the solver with a small, notebook-friendly API for
scoring a signature per cell and producing the Figure-3E-style plots in a couple
of calls (needs the `viz` extra: `pip install "nnpcapy[viz]"`).

```python
import emtscore as emt   # light import; plotting deps load lazily

# counts: DataFrame (cells x genes), raw counts; pt: pseudotime; time: labels
sE, _,     _   = emt.score_signature(counts, E_genes, sign_ref=-pt)          # epithelial
sM, load_M, gM = emt.score_signature(counts, M_genes, ncomp=2, sign_ref=pt)  # mesenchymal

scores = pd.DataFrame({"E score": sE[:, 0], "M score": sM[:, 0], "time": time})
fig, ax = emt.plot_em_scatter(scores, "E score", "M score",
                              group="time", ci=True)          # density + bootstrap-CI centroids

pc = emt.top_loading_genes(load_M, gM, n=5)                   # top genes of M PC1 / PC2
fig2 = emt.plot_signature_heatmap(counts, {"M PC1": pc[0], "M PC2": pc[1]},
                                  cell_group=time, group_order=order)
```

`score_signature` does total-library log-CPM + per-gene z-score + non-negative
sparse PCA; `sign_ref` orients each component (e.g. to pseudotime).

## Run the notebooks

```
pip install -e ".[full,notebooks]"    # notebooks extra adds jupyter + plotly
jupyter notebook notebooks/EMTscore_automated.ipynb
```

`EMTscore_automated.ipynb` is the slim version that calls into
`emtscore/workflow.py` for every step. `EMTscore_full_analysis.ipynb`
is the longer cell-by-cell exploration and additionally uses `plotly`
for one interactive figure. Both reproduce the same EMTscore analysis;
run either one top to bottom to see all figures.

## Run the benchmark

`paper/benchmark/` runs `nsprcomp` 705 times per side (47 matrices x 3
ncomps x 5 trials) on byte-identical `.npy` inputs, so the comparison
is implementation-only. (Install with the `benchmark` extra:
`pip install -e ".[benchmark,dev]"`.)

```
python paper/benchmark/datasets.py        # build cached inputs, ~5s
python paper/benchmark/bench_python.py    # ~3 min
# In RStudio: open paper/benchmark/bench_r.R and click Source, ~8 min
python paper/benchmark/compare.py         # merge + emit plots
```

The Python-only speedup ablation behind Fig 2A (no R required) lives in
`paper/benchmark/synth/`; see `REVIEW.md` section 8 for how to run it.

Latest numbers (shipping Python vs R nsprcomp 0.5.1-2; 47 matrices x 3
ncomps = 141 calls per side). All figures trace to
`paper/benchmark/results/summary.csv`.

- Total wall (sum of median call times): R 54.2s vs Python 2.8s -> ~19x.
- Per-call speedup: median 8.3x, max 49x.
- Single gene sets (Panchy E/M, 105-184 genes): median ~19x; biggest
  wins at ncomp >= 2 on the single-cell datasets.
- Multi gene sets (filtered C2 pathways, 8-93 genes): median ~8x.
- By data type: single-cell (n ~3.5-13k cells) median ~12x; bulk
  (n=120 cell lines) ~1.6x -- the covariance optimization needs large n
  to pay off (see paper/benchmark/README.md and Fig 2A).
- Memory: Python uses less heap throughout -- ~4-6x on single-cell
  single gene sets, ~22x on single-cell pathways, ~78x on bulk
  (median ~26x overall).

Honest framing of the speedup vs R: most of it is NumPy/BLAS
vectorization; the single algorithmic change beyond R is iterating the
EM on a precomputed covariance C = X^T X (O(n*d) -> O(d^2) per
iteration). Parameter defaults and the two-pass renormalization are R's
own -- reproduced, not novel. (Earlier versions of this file quoted
intermediate-stage benchmark numbers; those have been corrected.)

Plots are in `paper/benchmark/plots/`. `speedup_single.png` is the most
informative.

## What's in each subfolder

`src/nnpcapy/` is the installable library: `nsprcomp.py` is the EM
solver (NumPy port of the R algorithm), `nnpca.py` adds GMT parsing
and per-pathway helpers, `aucell.py` and `ssGSEA.py` are the two other
scoring methods the EMTscore family uses.

`emtscore/` is the application layer that uses the library: scoring
orchestration (`scoring.py`, `pipeline.py`), single-cell loaders
(`sc.py`, including `load_cook_adatas`), GMM clustering in E-M space,
pathway correlation, and three plotting modules (`plots_em.py`,
`plots_heatmap.py`, `plots_cook.py`) that reproduce the R vignette's
figures.

`utility/` has the Cook 2020 raw-data loader and the data-path
resolution helper used by the application.

`data/` bundles the Panchy E and M signatures, the bulk geneExp matrix
(120 cell lines x ~16k genes), the Cook 2020 single-cell em_expr CSVs
for A549 under EGF, TGF-beta, and TNF perturbations, the C2 EMT-related
pathway GMT, and the stemness / senescence TSV signatures.

## Acknowledgments

`nsprcomp` is by Sigg (2018); this is a NumPy translation
of the R implementation with modifications to improve efficiency.

MIT license.
