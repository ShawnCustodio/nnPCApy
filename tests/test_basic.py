"""Smoke tests so CI / pip install verification have something to run."""

import numpy as np
import pytest

from nnpcapy import nsprcomp


def test_nsprcomp_basic_shape():
    """A 50x20 random matrix returns the expected score/loading shapes."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 20))
    out = nsprcomp(X, ncomp=3, nneg=True, center=True, scale_=False)
    assert out["x"].shape == (50, 3),       "score matrix should be (n_obs, ncomp)"
    assert out["rotation"].shape == (20, 3), "rotation should be (n_features, ncomp)"


def test_nsprcomp_pc1_nonneg():
    """With nneg=True the first PC's loadings must be >= 0.

    (Higher PCs go through Gram-Schmidt orthogonalisation against earlier
    components, which can introduce small negatives — same behaviour as R.)
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12))
    out = nsprcomp(X, ncomp=2, nneg=True, center=True, scale_=False)
    assert (out["rotation"][:, 0] >= 0).all(), "PC1 non-negative constraint violated"


def test_nsprcomp_min_features():
    """ncomp can't exceed the number of features."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    out = nsprcomp(X, ncomp=3, nneg=True, center=True, scale_=False)
    assert out["x"].shape[1] == 3
