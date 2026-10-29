"""nsprcomp: non-negativity on every component, and agreement with R's nsprcomp.

The R agreement test feeds the solver R's own rnorm() stream, so Python and R start every
EM run from the same vector; a faithful port must then reproduce R's loadings and sdev to
floating-point precision on every component, not only PC1. The reference values in
tests/data/nsprcomp_r/ come from make_fixture.R there (R, nsprcomp 0.5.1-2).
"""
from pathlib import Path

import numpy as np
import pytest

from emtscore.nsprcomp import nsprcomp as emt_nsprcomp
from nnpcapy import nsprcomp

FIXTURE = Path(__file__).parent / "data" / "nsprcomp_r"


def _planted(seed, n=300, d=15):
    """n x d data with three planted non-negative sparse components plus noise."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for j in range(3):
        w = np.zeros(d)
        w[rng.choice(d, 8, replace=False)] = rng.random(8) + 0.2
        X += np.outer(rng.standard_normal(n) * 1.5 / (j + 1), w)
    return X + 0.5 * rng.standard_normal((n, d))


@pytest.mark.parametrize("seed", range(10))
def test_nonneg_on_every_component(seed):
    np.random.seed(seed)
    W = nsprcomp(_planted(seed), ncomp=4, nneg=True, center=True, scale_=False)["rotation"]
    assert (W >= 0).all()


R_CONFIGS = {
    "nneg": dict(nneg=True, scale_=False),
    "nneg_scale": dict(nneg=True, scale_=True),
    "dense": dict(nneg=False, scale_=False),
}


@pytest.mark.parametrize("cfg", R_CONFIGS)
def test_matches_r_with_matched_starts(monkeypatch, cfg):
    stream = iter(np.loadtxt(FIXTURE / "stream.csv"))
    monkeypatch.setattr(np.random, "randn", lambda d: np.fromiter(stream, float, count=d))
    X = np.loadtxt(FIXTURE / "X.csv", delimiter=",")

    out = nsprcomp(X, ncomp=3, center=True, **R_CONFIGS[cfg])

    np.testing.assert_allclose(out["rotation"], np.loadtxt(FIXTURE / f"rotation_{cfg}.csv", delimiter=","),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(out["sdev"], np.loadtxt(FIXTURE / f"sdev_{cfg}.csv"), rtol=1e-10)


def test_emtscore_uses_the_nnpcapy_solver():
    # one solver: emtscore re-exports nnpcapy's instead of keeping its own copy
    assert emt_nsprcomp is nsprcomp
