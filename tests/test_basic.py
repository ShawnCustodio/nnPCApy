"""Basic checks for nsprcomp."""

import numpy as np

from nnpcapy import nsprcomp


def test_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 20))
    out = nsprcomp(X, ncomp=3, nneg=True, center=True, scale_=False)
    assert out["x"].shape == (50, 3)
    assert out["rotation"].shape == (20, 3)


def test_pc1_nonneg():
    # PC1 has no orthogonalisation against earlier components, so
    # the non-negativity constraint should hold on its loadings.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12))
    out = nsprcomp(X, ncomp=2, nneg=True, center=True, scale_=False)
    assert (out["rotation"][:, 0] >= 0).all()


def test_min_features():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    out = nsprcomp(X, ncomp=3, nneg=True, center=True, scale_=False)
    assert out["x"].shape[1] == 3
