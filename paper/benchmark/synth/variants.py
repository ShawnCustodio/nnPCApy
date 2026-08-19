"""
Frozen nnPCA solver variants for the Panel 1 ablation.

The shipping library is NOT modified. Each variant below is a self-contained
function reproducing one historical optimisation stage, so the ablation
attributes the total speedup to individual mechanisms:

  V0  naive port        : data-matrix EM, single pass, nrestart=20, tol=1e-4, maxiter=200
  V1  + R-aligned params: data-matrix EM, single pass, nrestart=5,  tol=1e-3, maxiter=100
  V2  + two-pass support: data-matrix EM, two pass,     nrestart=5,  tol=1e-3, maxiter=100
  V3  + cov precompute  : shipping nnpcapy.nsprcomp (covariance EM, two pass, 5/1e-3/100)

V2 is algorithmically identical to upstream R nsprcomp (same params, same
two-pass "variational renormalization", same O(n*d) data-matrix EM), so V2-vs-R
isolates pure Python-vs-R overhead and V3-vs-V2 isolates the covariance trick.

All variants run with nneg=True, center=True, scale_=False (k=d, no sparsity
soft-thresholding) to match how nsprcomp is called in the gene-set pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# shipping library (V3)
_SRC = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()) / "src"
sys.path.insert(0, str(_SRC))
from nnpcapy import nsprcomp as _nsprcomp_shipping  # noqa: E402


# ----------------------------------------------------------------------
# Data-matrix EM (mirrors upstream R empca; omega=1, k=d so no soft-threshold)
# ----------------------------------------------------------------------
def _empca_data(Xp, Q, nneg, em_tol, em_maxiter):
    n, d = Xp.shape
    w = np.random.randn(d)
    if nneg:
        w = np.abs(w)
    w /= np.linalg.norm(w) + 1e-12

    obj = 0.0
    obj_old = -np.inf
    for _ in range(em_maxiter):
        z = Xp @ w                       # O(n*d)
        obj = float(z @ z)
        if obj != 0 and abs(obj - obj_old) / obj < em_tol:
            break
        obj_old = obj
        zz = z @ z
        w_star = (Xp.T @ z) / (zz if zz != 0 else 1e-12)   # O(n*d)
        if nneg:
            w_star[w_star < 0] = 0
        w = w_star
        if Q.shape[1] > 0:
            wo = Q.T @ w
            denom = np.sqrt(max(w @ w - wo @ wo, 1e-24))
            w = w / denom
        else:
            w = w / (np.linalg.norm(w) + 1e-12)
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, obj


def _empca_data_refined(Xp, Q, nneg, em_tol, em_maxiter):
    """Two-pass: find support, then re-run EM on the supported columns."""
    w, obj = _empca_data(Xp, Q, nneg, em_tol, em_maxiter)
    d = Xp.shape[1]
    supp = np.abs(w) > 0
    if supp.sum() == 0 or supp.sum() == d:
        return w, obj
    w_sub, obj = _empca_data(Xp[:, supp], Q[supp, :], nneg, em_tol, em_maxiter)
    w_out = np.zeros(d)
    w_out[supp] = w_sub
    return w_out, obj


def _run_data(X, ncomp, nneg, nrestart, em_tol, em_maxiter, refine):
    X = np.asarray(X, dtype=np.float64)
    cen = X.mean(axis=0)
    X = X - cen
    Xp = X.copy()
    n, d = X.shape
    W = np.zeros((d, ncomp))
    Q = np.zeros((d, ncomp))
    sdev = []
    em = _empca_data_refined if refine else _empca_data

    for cc in range(ncomp):
        best_obj, best_w = -np.inf, None
        for _ in range(nrestart):
            w, obj = em(Xp, Q[:, :cc], nneg, em_tol, em_maxiter)
            if obj > best_obj:
                best_obj, best_w = obj, w.copy()
        w = best_w
        W[:, cc] = w
        sdev.append(np.std(Xp @ w))
        if cc > 0:
            q = w - Q[:, :cc] @ (Q[:, :cc].T @ w)
        else:
            q = w.copy()
        nrm = np.linalg.norm(q)
        if nrm > 0:
            q /= nrm
        Q[:, cc] = q
        Xp -= np.outer(Xp @ q, q)
        if np.all(np.abs(Xp) < 1e-14):
            break

    return {"sdev": np.array(sdev), "rotation": W, "center": cen,
            "scale": np.ones(d), "x": X @ W}


# ----------------------------------------------------------------------
# Public variant callables: all take (X, ncomp) -> result dict
# ----------------------------------------------------------------------
def V0_naive(X, ncomp):
    return _run_data(X, ncomp, nneg=True, nrestart=20, em_tol=1e-4,
                     em_maxiter=200, refine=False)


def V1_params(X, ncomp):
    return _run_data(X, ncomp, nneg=True, nrestart=5, em_tol=1e-3,
                     em_maxiter=100, refine=False)


def V2_twopass(X, ncomp):
    return _run_data(X, ncomp, nneg=True, nrestart=5, em_tol=1e-3,
                     em_maxiter=100, refine=True)


def V3_shipping(X, ncomp):
    return _nsprcomp_shipping(X, ncomp=ncomp, nneg=True, center=True,
                              scale_=False)


VARIANTS = {
    "V0_naive": V0_naive,
    "V1_params": V1_params,
    "V2_twopass": V2_twopass,
    "V3_shipping": V3_shipping,
}
