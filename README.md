# nnPCApy

Python port of R's `nsprcomp` (non-negative sparse PCA), with a benchmark
comparing it against the original R implementation.

This was originally the nnPCA kernel inside an EMT scoring pipeline
([EMTScorePy](https://github.com/ShawnCustodio/EMTScorePy)). I pulled it
out so it can be installed and benchmarked on its own. The companion
methods (`aucell`, `ssGSEA`) are included because the EMTscore family
uses them alongside `nsprcomp`.

## Install

Local dev:

```
git clone https://github.com/ShawnCustodio/nnPCApy.git
cd nnPCApy
pip install -e ".[benchmark,dev]"
```

Direct from GitHub:

```
pip install git+https://github.com/ShawnCustodio/nnPCApy.git
```

(For the private repo you'll need an access token.)

## Use

```python
import numpy as np
from nnpcapy import nsprcomp

X = np.random.default_rng(0).standard_normal((100, 25))
out = nsprcomp(X, ncomp=2, nneg=True, center=True, scale_=False)

scores   = out["x"]         # (n_samples, ncomp)
loadings = out["rotation"]  # (n_features, ncomp), non-negative on PC1
```

The call signature mirrors R's `nsprcomp::nsprcomp(...)`.

## Benchmark

`benchmark/` runs `nsprcomp` 705 times on each side (47 input matrices x
3 component counts x 5 trials) using the same cached `.npy` inputs, so
the comparison is implementation-only and excludes I/O / preprocessing.

Numbers from the latest run on my hardware:

- Single gene set (Panchy E/M signatures, 12 cells of the sweep): median
  4.9x faster in Python, max 13x. The biggest wins are at `ncomp=2` on
  the single-cell datasets.
- Multi gene set (filtered C2 pathway sweep, 67 cells): median 1.2x
  faster. 24 of those cells are losses where R is faster on very small
  pathways, especially at `ncomp=3`.
- Total benchmark wall time: R 489s, Python 189s (2.6x).
- Heap on single-cell calls: median ~6x less in Python. The bulk
  comparison is dominated by R's session baseline so I excluded it from
  the headline.

Plots are in `benchmark/plots/`. `speedup_single.png` is the most
informative; `multi_total.png` shows the C2 sweep totals and
`memory_ratio.png` is the heap comparison.

To reproduce:

```
python benchmark/datasets.py        # build cached inputs, ~5s
python benchmark/bench_python.py    # ~3 min
# In RStudio: open benchmark/bench_r.R and click Source, ~8 min
python benchmark/compare.py         # merge + emit plots
```

`datasets.py` looks for the EMT data in this order:

1. `$NNPCAPY_DATA_DIR` if set,
2. `../EMTScorePy/data/` (sibling checkout - the default while EMTScorePy is the development driver),
3. `./data/` if you've put the data inside the repo.

If none exist, set `NNPCAPY_DATA_DIR` to a folder containing
`geneExp.csv`, `Panchy_et_al_{E,M}_signature.csv`, `filtered.c2.gmt`,
and `cook2020/A549_*_em_expr.csv`.

## What's in `src/nnpcapy/`

`nsprcomp.py` is the core EM-style solver, a direct port of the R
algorithm: alternate least-squares fit, non-negativity projection,
deflate against earlier components, normalise. `nnpca.py` adds GMT
parsing and per-pathway helpers used by EMTscore. `aucell.py` and
`ssGSEA.py` are the other two scoring methods.

Plotting, GMM clustering, and the single-cell loaders live in
[EMTScorePy](https://github.com/ShawnCustodio/EMTScorePy) - this repo
deliberately stays at the kernel level.

## Acknowledgments

`nsprcomp` is by Sigg and Buhmann (2008); this is a NumPy translation
of their R implementation. The EMTscore R package the work was
benchmarked against is at https://github.com/wenmm/EMTscore.

MIT license.
