# nnPCApy

Python port of R's `nsprcomp` (non-negative sparse PCA) and the EMT
scoring pipeline built on top of it, plus a benchmark comparing the
Python kernel against the original R implementation.

This repo started as the nnPCA kernel inside an EMT scoring analysis I
was doing. It now holds both pieces: the kernel as an installable
library, and the full EMTscore-style application that uses it.

## Layout

```
src/nnpcapy/    # installable library: nsprcomp + helpers
emtscore/       # EMT scoring application: scoring + plotting + GMM
utility/        # data path resolution, Cook 2020 raw-data loaders
data/           # bundled signatures, GMT files, bulk + Cook SC data
notebooks/      # two demo notebooks that reproduce the EMTscore vignette
benchmark/      # R-vs-Python timing harness + committed results/plots
tests/          # smoke tests for nsprcomp
EMTScorePy.html # rendered final report from the notebooks
```

## Install

For local development:

```
git clone https://github.com/ShawnCustodio/nnPCApy.git
cd nnPCApy
pip install -e ".[benchmark,dev]"
```

Once that's done, both `from nnpcapy import nsprcomp` (the library)
and `from emtscore import workflow as wf` (the application notebooks)
will work.

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

## Run the notebooks

```
jupyter notebook notebooks/EMTscore_automated.ipynb
```

`EMTscore_automated.ipynb` is the slim version that calls into
`emtscore/workflow.py` for every step. `EMTscore_full_analysis.ipynb`
is the longer cell-by-cell exploration. Both reproduce every figure in
`EMTScorePy.html`.

## Run the benchmark

`benchmark/` runs `nsprcomp` 705 times per side (47 matrices x 3
ncomps x 5 trials) on byte-identical `.npy` inputs, so the comparison
is implementation-only.

```
python benchmark/datasets.py        # build cached inputs, ~5s
python benchmark/bench_python.py    # ~3 min
# In RStudio: open benchmark/bench_r.R and click Source, ~8 min
python benchmark/compare.py         # merge + emit plots
```

Latest numbers (my hardware):

- Single gene set (Panchy E/M signatures): median 4.9x faster in Python,
  max 13x. Biggest wins at ncomp=2 on the single-cell datasets.
- Multi gene set (filtered C2 sweep): median 1.2x. R wins on 24 small
  pathways out of 108, mostly at ncomp=3.
- Total wall time: R 489s vs Python 189s.
- Memory on single-cell calls: median ~6x less heap in Python. Bulk
  heap comparison is dominated by R's session baseline, so I exclude
  it from the headline.

Plots are in `benchmark/plots/`. `speedup_single.png` is the most
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

`nsprcomp` is by Sigg and Buhmann (2008); this is a NumPy translation
of their R implementation. The EMTscore R package the work was
developed against is at https://github.com/wenmm/EMTscore.

MIT license.
