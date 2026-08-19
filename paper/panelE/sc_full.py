"""
Full-matrix single-cell loader + scoring (total-library log-CPM), used by
Panels B, C, D, E so every panel scores cells on the same genome-wide universe.

Data: panelE/cook_full/<dataset>_{counts.mtx,genes.txt,cells.txt,meta.csv}
(full Cook 2020 count matrix fetched via Bioconductor ExperimentHub).

Scoring for a gene set:
  1. total-library CPM within each cell (sum over ALL genes) and log1p,
  2. z-score each gene,
  3. nnPCA (non-negative sparse PCA, centered).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))
from nnpcapy.nsprcomp import nsprcomp  # noqa: E402

_DIR = Path(__file__).resolve().parent / "cook_full"
_CACHE: dict = {}


def load(dataset: str = "A549_TGFB1"):
    """Returns (X cells×genes float32, gene->col dict, genes list, per-cell library, meta)."""
    if dataset in _CACHE:
        return _CACHE[dataset]
    genes = [l.strip() for l in open(_DIR / f"{dataset}_genes.txt")]
    cells = [l.strip() for l in open(_DIR / f"{dataset}_cells.txt")]
    npz = _DIR / f"{dataset}_counts.npz"
    if npz.exists():                                            # compact shared form
        z = np.load(npz)
        C = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    else:                                                       # fall back to raw MatrixMarket
        C = mmread(_DIR / f"{dataset}_counts.mtx").tocsr()
    X = np.ascontiguousarray(C.toarray().T.astype(np.float32))  # cells × genes
    lib = X.sum(axis=1).astype(np.float64); lib[lib == 0] = 1.0
    meta = pd.read_csv(_DIR / f"{dataset}_meta.csv", index_col=0).reindex(cells)
    g2i = {g: i for i, g in enumerate(genes)}
    out = (X, g2i, genes, lib, meta)
    _CACHE[dataset] = out
    return out


def prep(counts_sub, lib_sub):
    """total-library log-CPM + per-gene z-score. counts_sub: cells×genes; lib_sub: cells."""
    Z = np.log1p(counts_sub / lib_sub[:, None] * 1e4)
    return (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)


def nnpca(counts_sub, lib_sub, ncomp=1):
    return nsprcomp(prep(counts_sub, lib_sub), ncomp=ncomp, nneg=True, center=True, scale_=False)


def logcpm_mean(counts_sub, lib_sub):
    """per-cell mean log-CPM over the gene set (for sign anchoring)."""
    return np.log1p(counts_sub / lib_sub[:, None] * 1e4).mean(axis=1)


def zc(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()
