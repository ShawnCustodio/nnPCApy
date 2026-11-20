This is a Python library with a companion manuscript under review.

nnPCApy: a Python library for non-negative principal component analysis and gene-set scoring

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
pip install nnpcapy          # PyPI, or:
pip install -e .             # from a clone -- just the nsprcomp solver
```

For the full toolkit (gene-set scoring, plotting, single-cell loaders):

```
pip install "nnpcapy[full]"          # PyPI, or from a clone:
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

# counts: DataFrame (cells x genes), raw counts; time: labels
sE, _,     _   = emt.score_signature(counts, E_genes)           # epithelial
sM, load_M, gM = emt.score_signature(counts, M_genes, ncomp=2)  # mesenchymal

scores = pd.DataFrame({"E score": sE[:, 0], "M score": sM[:, 0], "time": time})
fig, ax = emt.plot_em_scatter(scores, "E score", "M score",
                              group="time", ci=True)          # density + bootstrap-CI centroids

pc = emt.top_loading_genes(load_M, gM, n=5)                   # top genes of M PC1 / PC2
fig2 = emt.plot_signature_heatmap(counts, {"M PC1": pc[0], "M PC2": pc[1]},
                                  cell_group=time, group_order=order)
```

`score_signature` does total-library log-CPM + per-gene z-score + non-negative
sparse PCA. The loadings are non-negative, so each score increases with the
expression of its genes; no orientation step is needed.

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

`paper/benchmark/v2/` re-measures nnPCApy against R `nsprcomp` 0.5.1-2 on 47 inputs
(k = 1, 2 and 3), one process per fit, with BLAS threads pinned and the peak memory of
each fit read from the operating system. It also checks every component against R for
non-negativity and agreement. The full method, requirements, commands and run times are
in [`paper/benchmark/v2/README.md`](https://github.com/ShawnCustodio/nnPCApy/blob/main/paper/benchmark/v2/README.md).
The results are reported in the paper. Timings depend on the machine.

## What's in each subfolder

`src/nnpcapy/` is the installable library: `nsprcomp.py` is the core EM
solver, `nnpca.py` adds GMT parsing
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

`nsprcomp` is developed by Sigg (2018). nnPCApy is a NumPy translation
of the R implementation with modifications to improve efficiency.

MIT license.
