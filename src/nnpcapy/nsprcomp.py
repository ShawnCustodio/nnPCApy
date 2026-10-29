"""
Non-negative sparse PCA via EM (Sigg & Buhmann, ICML 2008): a port of R's nsprcomp.

Algorithm summary
-----------------
For each component cc:
  1. Precompute C = Xp.T @ Xp  (d×d covariance of the deflated data, once per component).
  2. Run `nrestart` independent EM restarts via `_empca_cov_refined`, keep the best.
     Each restart:
       a. First pass  — full EM on C to convergence; non-negativity zeroes part of the
                        features, revealing the active support S.
       b. Second pass — EM from a fresh start on C[S,S] (free array slice) for refined
                        weights within S. As in R, it also runs when S is every feature.
     Inside EM, w is rescaled against earlier components, never projected (see
     `_empca_cov`), so every component stays non-negative.
  3. Gram-Schmidt orthogonalize the winning w against previous components to get the
     deflation vector q (the loading w itself is kept as is).
  4. Deflate Xp in-place: Xp -= outer(Xp @ q, q).

Given the same random starts (R's rnorm() stream), this reproduces nsprcomp 0.5.1-2 to
floating-point precision on every component (tests/test_nsprcomp.py).

Differences from R
------------------
- Covariance precompute C = Xp.T @ Xp: O(d²) per EM iteration instead of O(n×d). It pays
  off when n is much larger than d (many cells, gene-set-sized d); see paper/benchmark/v2.
- With scale_=True, R stops on a constant column; here its scale is set to 1.
"""
import numpy as np


def _empca_cov(C, Q, nneg, em_tol, em_maxiter):
    """EM on precomputed covariance C = Xp.T @ Xp. O(d²) per iteration instead of O(n*d).

    Mirrors nsprcomp's internal empca(). After the non-negativity projection, w is only
    rescaled, so that its part orthogonal to the earlier deflation vectors Q has unit norm.
    It is not projected onto that orthogonal complement: projection would reintroduce
    negative entries, and the deflated C already ignores w's part in span(Q).
    """
    d = C.shape[0]
    w = np.random.randn(d)
    if nneg:
        w = np.abs(w)
    w /= np.linalg.norm(w)

    obj = 0.0
    obj_old = -np.inf
    for _ in range(em_maxiter):
        Cw = C @ w
        obj = float(w @ Cw)

        if obj == 0 or abs(obj - obj_old) / obj < em_tol:
            break
        obj_old = obj

        w = Cw / obj

        if nneg:
            w[w < 0] = 0

        if Q.shape[1] > 0:
            wo = Q.T @ w
            norm = np.sqrt(max(w @ w - wo @ wo, 0.0))
        else:
            norm = np.linalg.norm(w)
        if norm < 1e-12:
            break
        w /= norm

    norm = np.linalg.norm(w)
    return (w / norm if norm > 0 else w), obj


def _empca_cov_refined(C, Q, nneg, em_tol, em_maxiter):
    """Two-pass EM, as in nsprcomp: the first pass finds the support S = {j : w_j != 0},
    the second re-solves from a fresh random start on C[S, S] (a free slice of C).
    Like R, the second pass always runs, also when S is every feature."""
    w, obj = _empca_cov(C, Q, nneg, em_tol, em_maxiter)
    supp = np.flatnonzero(w)
    d = C.shape[0]
    if len(supp) == 0:
        return w, obj
    if len(supp) == d:
        return _empca_cov(C, Q, nneg, em_tol, em_maxiter)

    C_sub = C[np.ix_(supp, supp)]
    w_sub, obj = _empca_cov(C_sub, Q[supp, :], nneg, em_tol, em_maxiter)
    w_out = np.zeros(d)
    w_out[supp] = w_sub
    return w_out, obj


def nsprcomp(
    x,
    ncomp=1,
    center=True,
    scale_=False,
    nneg=True,
    nrestart=5,
    em_tol=1e-3,
    em_maxiter=100,
):
    """Non-negative sparse PCA via expectation-maximization (Sigg & Buhmann, 2008).

    A drop-in Python port of R's ``nsprcomp::nsprcomp()`` -- same defaults,
    same algorithm (see the module docstring above for the EM summary).

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Data matrix, samples (e.g. cells) in rows and features (e.g. genes) in
        columns.
    ncomp : int, default 1
        Number of components to extract, largest-variance first.
    center : bool, default True
        Mean-center each column before fitting.
    scale_ : bool, default False
        Scale each column to unit variance before fitting (R's ``scale.``).
    nneg : bool, default True
        Constrain loadings to be non-negative. This is what makes each axis
        directly interpretable as a non-negative combination of features;
        set False to recover ordinary PCA.
    nrestart : int, default 5
        Number of random EM restarts per component; the highest-variance
        solution is kept. EM only finds a local optimum, so more restarts
        trade runtime for a better chance of finding the global one.
    em_tol : float, default 1e-3
        Relative convergence tolerance for the EM objective.
    em_maxiter : int, default 100
        Maximum EM iterations per restart.

    Returns
    -------
    dict with keys:
        x : ndarray, shape (n_samples, ncomp)
            Per-sample scores (the data projected onto each component).
        rotation : ndarray, shape (n_features, ncomp)
            Loadings; non-negative on every component when ``nneg=True``.
        sdev : ndarray, shape (ncomp,)
            Standard deviation explained by each component.
        center : ndarray, shape (n_features,)
            Column means subtracted (zeros if ``center=False``).
        scale : ndarray, shape (n_features,)
            Column scale factors applied (ones if ``scale_=False``).
    """
    X = np.asarray(x, dtype=np.float64)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    if center:
        cen = X.mean(axis=0)
        X = X - cen
    else:
        cen = np.zeros(X.shape[1])

    if scale_:
        # R's scale.: root-mean-square with n - 1, which is the sd when X is centred
        sc = np.sqrt((X * X).sum(axis=0) / (X.shape[0] - 1))
        sc[sc == 0] = 1
        X = X / sc
    else:
        sc = np.ones(X.shape[1])

    Xp = X.copy()
    _n, d = X.shape
    W = np.zeros((d, ncomp))
    Q = np.zeros((d, ncomp))
    sdev = []

    for cc in range(ncomp):
        # Precompute covariance once per component: O(n*d²) up front, then O(d²) per EM iter
        C = Xp.T @ Xp

        best_obj = -np.inf
        best_w = None

        for _ in range(nrestart):
            w, obj = _empca_cov_refined(C, Q[:, :cc], nneg, em_tol, em_maxiter)
            if obj > best_obj:
                best_obj = obj
                best_w = w.copy()

        w = best_w
        W[:, cc] = w
        sdev.append(np.std(Xp @ w, ddof=1))   # R: sd(), divisor n - 1

        if cc > 0:
            q = w - Q[:, :cc] @ (Q[:, :cc].T @ w)
        else:
            q = w.copy()

        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        Q[:, cc] = q
        Xp -= np.outer(Xp @ q, q)

        if np.all(np.abs(Xp) < 1e-14):
            break

    return {
        "sdev": np.array(sdev),
        "rotation": W,
        "center": cen,
        "scale": sc,
        "x": X @ W,
    }
