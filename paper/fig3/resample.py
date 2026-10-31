"""Shared resampling for the Fig 3 and Fig S2 uncertainty analyses.

Three arms separate the two sources of variation in a refitted nnPCA statistic:

  seed      the full data, refitted from a new set of random starts each time
  sampling  resampled cells, refitted from one fixed set of random starts
  both      resampled cells and new random starts each time (the usual bootstrap)

"sampling" and "both" draw the same cells for the same b, so they can be compared draw by
draw. Fixed starts are possible because nsprcomp(random_state=...) gives every restart the
same starting vector on any data with the same genes (src/nnpcapy/nsprcomp.py).

Cells are resampled with replacement within strata, by default time point x culture: Cook
et al. (2020) ran the time course twice with two plates each, so Mix1, Mix2, Mix3 and Mix4
are independent cultures, and Mix3a/b, Mix4a/b are two 10x libraries of one plate. The
intervals are therefore conditional on these cultures. Intervals are percentile intervals.

Refits use NRESTART = 20 random restarts per component instead of nsprcomp's default 5,
which often stops at a weaker solution for some gene sets (pc_order.py, Fig S2);
full-data reference fits use REF_NRESTART = 50.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = (Path(os.environ["NNPCAPY_REPO"]) if "NNPCAPY_REPO" in os.environ
        else next(p for p in HERE.parents if (p / "pyproject.toml").exists()))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper" / "panelE"))
import sc_full  # noqa: E402

from nnpcapy.nsprcomp import LocalOptimumWarning, nsprcomp  # noqa: E402

DATA = REPO / "data"
RESULTS = HERE / "results"
ARMS = ("seed", "sampling", "both")
SEED = 20261029
NRESTART = 20
REF_NRESTART = 50


def load(dataset: str = "A549_TGFB1"):
    """Cook 2020 counts: (cells x genes, gene -> column, genes, library sizes, metadata)."""
    return sc_full.load(dataset)


def zscores(counts: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """Library-size log-CPM, z-scored per gene, as for every Fig 3 score (sc_full.prep)."""
    return sc_full.prep(counts, lib)


def parse_gmt(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            out[p[0]] = [g for g in p[2:] if g]
    return out


def cultures(meta) -> np.ndarray:
    """Independent culture of each cell: Mix3a/b and Mix4a/b are libraries of one plate."""
    return meta["Mix"].astype(str).str.replace(r"[ab]$", "", regex=True).to_numpy()


def strata(*labels) -> list[np.ndarray]:
    """Row indices of each combination of the given per-cell labels."""
    keys = np.array(["|".join(map(str, t)) for t in zip(*labels)])
    return [np.flatnonzero(keys == k) for k in np.unique(keys)]


def resample_rows(n: int, rng: np.random.Generator, groups: list[np.ndarray] | None = None) -> np.ndarray:
    """n rows drawn with replacement, within each group if groups are given."""
    if groups is None:
        return rng.integers(0, n, n)
    return np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


def draws(arm: str, n: int, B: int, seed: int = SEED, groups=None):
    """Yield (b, rows, start) for B draws of one arm. ``start(k)`` is the random_state for
    the k-th fit of the draw, so separate gene sets get separate streams."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    for b in range(B):
        rows = np.arange(n) if arm == "seed" else resample_rows(n, np.random.default_rng([seed, 2, b]), groups)
        key = {"seed": [seed, 1, b], "sampling": [seed, 1, 0], "both": [seed, 3, b]}[arm]
        yield b, rows, (lambda k, key=key: np.random.SeedSequence([*key, k]))


def fit(Z: np.ndarray, ncomp: int, random_state, nrestart: int = NRESTART) -> dict:
    """nnPCA (centred, non-negative) of a cells x genes block, usually z-scored by zscores().
    LocalOptimumWarning is silenced: callers read the result's "suboptimal" flags instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LocalOptimumWarning)
        return nsprcomp(Z, ncomp=ncomp, nneg=True, center=True, scale_=False,
                        nrestart=nrestart, random_state=random_state)


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else abs(float(a @ b)) / (na * nb)


def match2(W: np.ndarray, Wref: np.ndarray) -> tuple[int, int]:
    """Columns of W matched to reference PC1 and PC2: (0, 1) unless swapping fits better
    by summed |cos| of the loadings."""
    ident = abs_cos(W[:, 0], Wref[:, 0]) + abs_cos(W[:, 1], Wref[:, 1])
    swap = abs_cos(W[:, 0], Wref[:, 1]) + abs_cos(W[:, 1], Wref[:, 0])
    return (0, 1) if ident >= swap else (1, 0)


def max_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Largest principal angle between the column spans of A and B (90 if either is rank-deficient)."""
    Qa, Ra = np.linalg.qr(A)
    Qb, Rb = np.linalg.qr(B)
    if min(np.abs(np.diag(Ra)).min(), np.abs(np.diag(Rb)).min()) < 1e-12:
        return 90.0
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    return float(np.degrees(np.arccos(s.min())))


def ci(x, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(x, [a, 100 - a])
    return float(lo), float(hi)


def ci_mc_se(x, level: float = 0.95, reps: int = 200, seed: int = 0) -> tuple[float, float]:
    """Monte Carlo standard error of each interval endpoint, by resampling the B draws."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    ends = np.array([ci(x[rng.integers(0, len(x), len(x))], level) for _ in range(reps)])
    return float(ends[:, 0].std()), float(ends[:, 1].std())


def support_se(p: float, B: int) -> float:
    """Monte Carlo standard error of a support frequency p estimated from B draws."""
    return float(np.sqrt(p * (1 - p) / B))
