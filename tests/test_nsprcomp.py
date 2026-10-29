"""nsprcomp: non-negativity on every component, and agreement with R's nsprcomp.

The R agreement test feeds the solver R's own rnorm() stream, so Python and R start every
EM run from the same vector; a faithful port must then reproduce R's loadings and sdev to
floating-point precision on every component, not only PC1. The reference values in
tests/data/nsprcomp_r/ come from make_fixture.R there (R, nsprcomp 0.5.1-2).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emtscore.nnpca import get_nnPCA_result as emt_get_nnPCA_result
from emtscore.nsprcomp import nsprcomp as emt_nsprcomp
from nnpcapy import nsprcomp
from nnpcapy.nnpca import get_nnPCA_result

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


@pytest.mark.parametrize("score", [get_nnPCA_result, emt_get_nnPCA_result], ids=["src", "emtscore"])
def test_gene_set_score_increases_with_expression(score):
    X = _planted(0, n=200, d=12)
    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(12)])
    genes = list(df.columns)
    for seed in range(5):
        np.random.seed(seed)
        s = score(df, genes)
        assert np.corrcoef(s, df[genes].mean(axis=1))[0, 1] > 0
        np.random.seed(seed)
        np.testing.assert_allclose(score(df, genes, align_direction="negative"), -s)


def test_score_signature_follows_expression():
    # no orientation step: with non-negative loadings every component's score rises with
    # the expression of the genes that component uses
    from emtscore.single_cell import score_signature
    rng = np.random.default_rng(3)
    counts = rng.poisson(5 * np.exp(_planted(1, n=200, d=15) / 3)).astype(float)
    genes = [f"g{i}" for i in range(15)]
    df = pd.DataFrame(counts, columns=genes)
    for seed in range(3):
        np.random.seed(seed)
        scores, load, _used = score_signature(df, genes, ncomp=2)
        assert (load >= 0).all()
        logcpm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
        for k in range(2):
            on = load[:, k] > 0
            assert np.corrcoef(scores[:, k], logcpm[:, on].mean(1))[0, 1] > 0


def test_em_scores_are_not_oriented_by_pseudotime():
    # E genes that *rise* along pseudotime must give an E score that rises with them; the
    # pipeline used to flip the E score whenever it correlated positively with pseudotime
    from types import SimpleNamespace

    from emtscore.pipeline import compute_em_scores
    rng = np.random.default_rng(5)
    n = 300
    pt = np.sort(rng.random(n))
    E = rng.poisson(np.exp(1 + 2 * pt[:, None] + 0.3 * rng.standard_normal((n, 8))))
    M = rng.poisson(np.exp(1 + 0.5 * rng.standard_normal((n, 8))))
    genes = [f"E{i}" for i in range(8)] + [f"M{i}" for i in range(8)]
    adata = SimpleNamespace(X=np.hstack([E, M]).astype(float), obs=pd.DataFrame({"Pseudotime": pt}),
                            obs_names=pd.RangeIndex(n), var_names=pd.Index(genes))
    np.random.seed(0)
    Es, _, _ = compute_em_scores(adata, genes[:8], genes[8:], ncomp_M=1)
    assert np.corrcoef(Es, pt)[0, 1] > 0
