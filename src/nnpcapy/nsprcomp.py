"""
Non-negative sparse PCA via EM (Sigg & Buhmann, ICML 2008): a port of R's nsprcomp.

Algorithm summary
-----------------
For each component cc:
  1. Run `nrestart` independent EM restarts via `_empca_refined`, keep the best.
     Each restart:
       a. First pass  — full EM to convergence; non-negativity zeroes part of the
                        features, revealing the active support S.
       b. Second pass — EM from a fresh start on the features in S, for refined
                        weights within S. As in R, it also runs when S is every feature.
     Inside EM, w is rescaled against earlier components, never projected (see
     `_empca`), so every component stays non-negative.
  2. Gram-Schmidt orthogonalize the winning w against previous components to get the
     deflation vector q (the loading w itself is kept as is).
  3. Deflate: Xp <- Xp (I - q q').

Given the same random starts (R's rnorm() stream), this reproduces nsprcomp 0.5.1-2 to
floating-point precision on every component (tests/test_nsprcomp.py).

Two ways to compute the same EM
-------------------------------
Each EM step needs Xp' Xp w for the deflated data Xp.
- method="cov": form C = Xp' Xp once (d×d) and iterate w <- C w, O(d²) per step instead
  of O(n×d); after each component C is deflated in place, C <- (I - qq') C (I - qq').
  Pays off when n is much larger than d (many cells, gene-set-sized d).
- method="data": work on X itself, as R does, with centring, scaling and deflation applied
  implicitly. Sparse input (any scipy.sparse matrix, e.g. AnnData's .X) is never made
  dense, so memory stays near the size of X. Pays off when d is large (whole
  transcriptome) or close to n.
- method="auto" (default) picks the cheaper one per EM step; both draw the same random
  numbers and give the same result.

Differences from R
------------------
- With scale_=True, R stops on a constant column; here its scale is set to 1.
- R stops early when every entry of the deflated data is below 1e-14; here when the
  remaining variance is below 1e-10 of the total. Both only trigger when ncomp exceeds
  the rank of the data.
"""
import numpy as np

_STOP_RTOL = 1e-10          # stop when the deflated data keeps less variance than this
_COV_MAX_D = 4096           # largest d for which method="auto" forms the d×d matrix C
_BLOCK = 1024               # rows per block when forming or deflating C


def _is_sparse(x):
    # scipy.sparse matrices and arrays, without importing scipy
    return hasattr(x, "tocsr") and hasattr(x, "nnz")


def _empca(gram, Q, d, nneg, em_tol, em_maxiter):
    """One EM run; ``gram(w)`` returns (Xp' Xp w, ||Xp w||²) for the current problem.

    Mirrors nsprcomp's internal empca(). After the non-negativity projection, w is only
    rescaled, so that its part orthogonal to the earlier deflation vectors Q has unit norm.
    It is not projected onto that orthogonal complement: projection would reintroduce
    negative entries, and the deflated data already ignore w's part in span(Q).
    """
    w = np.random.randn(d)
    if nneg:
        w = np.abs(w)
    w /= np.linalg.norm(w)

    obj = 0.0
    obj_old = -np.inf
    for _ in range(em_maxiter):
        Cw, obj = gram(w)

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


def _empca_refined(gram, sub, d, Q, nneg, em_tol, em_maxiter):
    """Two-pass EM, as in nsprcomp: the first pass finds the support S = {j : w_j != 0},
    the second re-solves from a fresh random start on the features in S (``sub(S)``).
    Like R, the second pass always runs, also when S is every feature."""
    w, obj = _empca(gram, Q, d, nneg, em_tol, em_maxiter)
    supp = np.flatnonzero(w)
    if len(supp) == 0:
        return w, obj
    if len(supp) == d:
        return _empca(gram, Q, d, nneg, em_tol, em_maxiter)

    w_sub, obj = _empca(sub(supp), Q[supp, :], len(supp), nneg, em_tol, em_maxiter)
    w_out = np.zeros(d)
    w_out[supp] = w_sub
    return w_out, obj


class _CovEM:
    """method="cov": EM on C = Xp' Xp, deflated in place after each component."""

    def __init__(self, Xs, sparse, cen, sc):
        d = Xs.shape[1]
        if sparse:
            # centre and scale one block of rows at a time: never a dense copy of X
            self.C = np.zeros((d, d))
            for i in range(0, Xs.shape[0], _BLOCK):
                B = (Xs[i:i + _BLOCK].toarray() - cen) / sc
                self.C += B.T @ B
        else:
            self.C = Xs.T @ Xs

    def gram(self, w):
        Cw = self.C @ w
        return Cw, float(w @ Cw)

    def sub(self, S):
        C_S = self.C[np.ix_(S, S)]

        def gram_S(w):
            Cw = C_S @ w
            return Cw, float(w @ Cw)
        return gram_S

    def residual(self):
        return float(np.trace(self.C))

    def deflate(self, q):
        # C <- (I - qq') C (I - qq') = C - c q' - q (c - s q)',  c = C q, s = q'c
        c = self.C @ q
        s = float(q @ c)
        r = c - s * q
        for i in range(0, len(q), _BLOCK):
            blk = slice(i, i + _BLOCK)
            self.C[blk] -= np.outer(c[blk], q) + np.outer(q[blk], r)


class _DataEM:
    """method="data": EM on X itself, Xp = ((X - 1 cen') / sc)(I - Q Q'), applied implicitly."""

    def __init__(self, Xs, sparse, cen, sc, total):
        self.X, self.sparse, self.cen, self.sc = Xs, sparse, cen, sc
        self.d = Xs.shape[1]
        self.Q = np.zeros((self.d, 0))
        self.left = total            # ||Xp||_F², updated as components are removed

    def _mv(self, v):                # Xp v
        u = v - self.Q @ (self.Q.T @ v)
        if not self.sparse:
            return self.X @ u        # dense X is already centred and scaled
        u = u / self.sc
        return self.X @ u - float(self.cen @ u)

    def _rmv(self, z):               # Xp' z
        if self.sparse:
            t = (self.X.T @ z - self.cen * z.sum()) / self.sc
        else:
            t = self.X.T @ z
        return t - self.Q @ (self.Q.T @ t)

    def gram(self, w):
        z = self._mv(w)
        return self._rmv(z), float(z @ z)

    def sub(self, S):
        def gram_S(w_S):
            w = np.zeros(self.d)
            w[S] = w_S
            z = self._mv(w)
            return self._rmv(z)[S], float(z @ z)
        return gram_S

    def residual(self):
        return self.left

    def deflate(self, q):
        z = self._mv(q)              # Xp q, before q joins Q
        self.left -= float(z @ z)
        self.Q = np.column_stack([self.Q, q])


def _choose_method(n, d, nnz):
    """Per EM step, "cov" costs about d² and "data" about 2 nnz (nnz = n d if dense)."""
    return "cov" if d <= _COV_MAX_D and d * d <= 2 * nnz else "data"


def nsprcomp(
    x,
    ncomp=1,
    center=True,
    scale_=False,
    nneg=True,
    nrestart=5,
    em_tol=1e-3,
    em_maxiter=100,
    method="auto",
):
    """Non-negative sparse PCA via expectation-maximization (Sigg & Buhmann, 2008).

    A drop-in Python port of R's ``nsprcomp::nsprcomp()`` -- same defaults,
    same algorithm (see the module docstring above for the EM summary).

    Parameters
    ----------
    x : array-like or scipy.sparse matrix, shape (n_samples, n_features)
        Data matrix, samples (e.g. cells) in rows and features (e.g. genes) in
        columns. Sparse input (e.g. AnnData's ``.X``) is not made dense.
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
    method : {"auto", "cov", "data"}, default "auto"
        How each EM step is computed (see the module docstring). Both give the
        same result; "auto" picks the cheaper one for the shape of ``x``.

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
        method : str
            "cov" or "data", the way the EM was computed.
    """
    sparse = _is_sparse(x)
    if sparse:
        X = x.tocsr().astype(np.float64)
    else:
        X = np.asarray(x, dtype=np.float64)
    n, d = X.shape

    if center:
        cen = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    else:
        cen = np.zeros(d)

    if sparse:
        # column sums of squares of X - 1 cen', without centring X
        ss = np.asarray(X.multiply(X).sum(axis=0)).ravel() - n * cen * cen
        ss = np.maximum(ss, 0.0)
    else:
        X = X - cen                      # one centred copy; the input is never modified
        ss = (X * X).sum(axis=0)

    if scale_:
        # R's scale.: root-mean-square with n - 1, which is the sd when X is centred
        sc = np.sqrt(ss / (n - 1))
        sc[sc == 0] = 1
        ss = ss / (sc * sc)
        if not sparse:
            X = X / sc
    else:
        sc = np.ones(d)

    if method == "auto":
        method = _choose_method(n, d, X.nnz if sparse else n * d)
    if method == "cov":
        em = _CovEM(X, sparse, cen, sc)
    elif method == "data":
        em = _DataEM(X, sparse, cen, sc, float(ss.sum()))
    else:
        raise ValueError(f"method must be 'auto', 'cov' or 'data', not {method!r}")

    # column means of the scaled data, deflated alongside it (zero when center=True)
    if center:
        mean = np.zeros(d)
    else:
        mean = np.asarray(X.mean(axis=0), dtype=np.float64).ravel() / (sc if sparse else 1)
    total = em.residual()

    W = np.zeros((d, ncomp))
    Q = np.zeros((d, ncomp))
    sdev = []

    for cc in range(ncomp):
        best_obj = -np.inf
        best_w = None

        for _ in range(nrestart):
            w, obj = _empca_refined(em.gram, em.sub, d, Q[:, :cc], nneg, em_tol, em_maxiter)
            if obj > best_obj:
                best_obj = obj
                best_w = w.copy()

        w = best_w
        W[:, cc] = w
        # R: sd(Xp %*% w), divisor n - 1
        _, ssq = em.gram(w)
        sdev.append(np.sqrt(max(ssq - n * float(mean @ w) ** 2, 0.0) / (n - 1)))

        if cc > 0:
            q = w - Q[:, :cc] @ (Q[:, :cc].T @ w)
        else:
            q = w.copy()

        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        Q[:, cc] = q
        em.deflate(q)
        mean -= float(mean @ q) * q

        if em.residual() <= _STOP_RTOL * total:
            break

    if sparse:
        Ws = W / sc[:, None]
        scores = np.asarray(X @ Ws) - cen @ Ws
    else:
        scores = X @ W

    return {
        "sdev": np.array(sdev),
        "rotation": W,
        "center": cen,
        "scale": sc,
        "x": scores,
        "method": method,
    }
