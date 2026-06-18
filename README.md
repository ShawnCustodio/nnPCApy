# nnPCApy

A pure-Python port of R's `nsprcomp` (non-negative sparse PCA) plus the
benchmark that backs the claim *"this is faster than R."*

`nnPCApy` is the kernel underlying the EMTscore family of gene-signature
scoring methods, lifted out of its original domain-specific R package
and rewritten in NumPy so it can be installed, imported, and benchmarked
as a standalone library. The companion methods (AUCell, ssGSEA) are
included for completeness — they're what the EMTscore reference pipeline
calls alongside `nsprcomp`.

The repo is split into three things:

- `src/nnpcapy/` — the importable package
- `benchmark/` — the R-vs-Python timing harness and committed results
- `tests/` — minimal smoke tests

## Install

Once the repo is public (or with an access token for a private one):

```bash
pip install git+https://github.com/ShawnCustodio/nnPCApy.git
```

Local development:

```bash
git clone https://github.com/ShawnCustodio/nnPCApy.git
cd nnPCApy
pip install -e ".[benchmark,dev]"
```

## Usage

```python
import numpy as np
from nnpcapy import nsprcomp

# X: (n_samples, n_features) — e.g. (cells, signature genes)
X = np.random.default_rng(0).standard_normal((100, 25))

out = nsprcomp(X, ncomp=2, nneg=True, center=True, scale_=False)

scores   = out["x"]         # (100, 2)   per-sample scores along each PC
loadings = out["rotation"]  # (25, 2)    per-gene non-negative weights
```

The signature mirrors R's `nsprcomp::nsprcomp(...)` exactly, so anyone
familiar with the R original can drop straight in.

## Benchmark — Python vs R

The benchmark calls `nsprcomp` 705 times (47 distinct input matrices × 3
component counts × 5 trials + warm-up) on both sides and compares wall
time and peak heap. The two sides read **byte-identical** cached `.npy`
inputs, so the speedup number reflects implementation only — not I/O,
parsing, or preprocessing differences.

### Headline numbers (this run)

| Slice | Median speedup | Range |
|---|---:|---|
| Single gene set (Panchy E / M, 12 cells) | **4.88×** | 2.93× to 13.12× |
| Multi gene set (filtered C2, 67 cells) | 1.22× | 0.31× to 7.55× |
| Overall (141 cells) | 1.50× | — |
| Total benchmark wall time | **2.6× faster** | R 489 s vs Python 189 s |

Memory (single-cell datasets only, where the comparison is algorithmically
honest): Python uses a median **6.2× less heap** than R on single-gene-set
calls.

### What the plots show

`benchmark/plots/speedup_single.png` — per-condition Python-vs-R speedup
for the single-gene-set workload. Bars above 1× mean Python wins. SC
datasets typically 4-13×; bulk (120 samples) is roughly tied.

`benchmark/plots/multi_total.png` — total time across the 12-pathway C2
sweep per dataset. Python wins on every dataset × ncomp combination.

`benchmark/plots/memory_ratio.png` — peak-heap ratio. SC values 3-15×
are honest algorithmic differences; bulk values 80-150× are dominated by
R's session-baseline overhead, not the kernel.

### Reproducing the numbers

```bash
# 1. Build cached input matrices from EMTScorePy data
python benchmark/datasets.py

# 2. Python timings
python benchmark/bench_python.py

# 3. R timings (in RStudio, open benchmark/bench_r.R and click Source)
#    Requires: install.packages(c("nsprcomp", "RcppCNPy"))

# 4. Merge + plot
python benchmark/compare.py
```

`benchmark/datasets.py` looks for benchmark data via:

1. Environment variable `NNPCAPY_DATA_DIR` (explicit override), then
2. `../EMTScorePy/data/` (sibling checkout — the default for this repo), then
3. `./data/` (bundled — if a user has placed data inside nnPCApy itself).

If none are found, set `NNPCAPY_DATA_DIR` to a folder containing
`geneExp.csv`, `Panchy_et_al_{E,M}_signature.csv`, `filtered.c2.gmt`,
and `cook2020/A549_*_em_expr.csv`.

## What's in `src/nnpcapy/`

`nsprcomp.py` — the core non-negative sparse PCA solver. The EM-style
iteration (alternating least-squares fit + non-negativity projection +
deflation) is a direct port of the R algorithm. Public API:
`nsprcomp(X, ncomp, nneg, center, scale_)`.

`nnpca.py` — convenience wrappers including GMT parsing and per-pathway
PCA helpers, used by the higher-level EMTscore notebooks.

`aucell.py` — rank-AUC pathway scoring (single-cell-friendly,
normalisation-robust).

`ssGSEA.py` — single-sample gene-set enrichment analysis.

The package is intentionally narrow. It does *not* include the
EMTscore-specific plotting, GMM clustering, or single-cell loaders —
those live in the application layer ([EMTScorePy](https://github.com/ShawnCustodio/EMTScorePy)).

## Provenance and prior art

`nsprcomp.py` is a faithful reimplementation of the R `nsprcomp` package
(Sigg & Buhmann, 2008). The benchmark exists to substantiate the speed
claim of the rewrite; it is not introducing a new scoring method.

The EMTscore reference pipeline this work was originally developed
against is at https://github.com/wenmm/EMTscore. The downstream
application layer using `nnpcapy` against EMT data is at
https://github.com/ShawnCustodio/EMTScorePy.

## License

MIT — see [LICENSE](LICENSE).
