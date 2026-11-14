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

Random starts
-------------
With random_state=None (default) the starts come from NumPy's global generator in R's
order, one np.random.randn() call per EM run. With random_state set, each restart draws
from its own stream, keyed by the seed, the component and the restart: the fit is
reproducible, the global generator is untouched, and data with the same number of
features get the same starting vectors (used to separate the effect of the random
starts from that of resampling the data).

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
- A check R does not make: if component k+1, which is non-negative and so a candidate for
  component k too, explains more of the data left before component k than component k
  itself, then component k is a local optimum of the EM and a better solution exists.
  Such components are marked in the result ("suboptimal") and a LocalOptimumWarning is
  issued. The check is sufficient, not necessary: it cannot see a better solution that no
  later component happens to point at, and with ncomp=1 it never applies.
"""
import warnings

import numpy as np

_STOP_RTOL = 1e-10          # stop when the deflated data keeps less variance than this
_COV_MAX_D = 4096           # largest d for which method="auto" forms the d×d matrix C
_BLOCK = 1024               # rows per block when forming or deflating C


class LocalOptimumWarning(UserWarning):
    """A component is a local optimum of the EM: a better non-negative solution exists."""


def _is_sparse(x):
    # scipy.sparse matrices and arrays, without importing scipy
    return hasattr(x, "tocsr") and hasattr(x, "nnz")


def _empca(gram, Q, w, nneg, em_tol, em_maxiter):
    """One EM run from the start w (standard normal draws); ``gram(w)`` returns
    (Xp' Xp w, ||Xp w||²) for the current problem.

    Mirrors nsprcomp's internal empca(). After the non-negativity projection, w is only
    rescaled, so that its part orthogonal to the earlier deflation vectors Q has unit norm.
    It is not projected onto that orthogonal complement: projection would reintroduce
    negative entries, and the deflated data already ignore w's part in span(Q).
    """
    w = np.abs(w) if nneg else np.array(w, dtype=np.float64)
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


def _global_starts(d, S=None):
    # R's order: one draw per EM run, of the length of that run's problem
    return np.random.randn(d if S is None else len(S))


def _keyed_starts(seed, cc, r):
    """Starts for restart r of component cc from their own stream: a full-length draw
    per pass, restricted to S on the second, so they depend only on (seed, cc, r, d)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, cc, r)))

    def starts(d, S=None):
        w = rng.standard_normal(d)
        return w if S is None else w[S]
    return starts


def _empca_refined(gram, sub, d, Q, nneg, em_tol, em_maxiter, starts):
    """Two-pass EM, as in nsprcomp: the first pass finds the support S = {j : w_j != 0},
    the second re-solves from a fresh random start on the features in S (``sub(S)``).
    Like R, the second pass always runs, also when S is every feature."""
    w, obj = _empca(gram, Q, starts(d), nneg, em_tol, em_maxiter)
    supp = np.flatnonzero(w)
    if len(supp) == 0:
        return w, obj
    if len(supp) == d:
        return _empca(gram, Q, starts(d), nneg, em_tol, em_maxiter)

    w_sub, obj = _empca(sub(supp), Q[supp, :], starts(d, supp), nneg, em_tol, em_maxiter)
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
        return c, s


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
        c, s = self._rmv(z), float(z @ z)
        self.left -= s
        self.Q = np.column_stack([self.Q, q])
        return c, s


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
    random_state=None,
):
    """Non-negative sparse PCA via expectation-maximization (Sigg & Buhmann, 2008).

    A Python port of R's ``nsprcomp::nsprcomp()`` with the same defaults and the same EM.
    Given the same random starts it reproduces R's loadings, and every component is
    non-negative. It also accepts sparse input and ``random_state``, and warns about local
    optima R does not detect (see "Differences from R" in the module docstring).

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
    random_state : None, int or numpy.random.SeedSequence, default None
        Seed for the random starts. None uses NumPy's global generator, as R uses
        its own. With a seed the fit is reproducible and does not touch the global
        generator, and every restart gets the same starting vector on any data with
        the same number of features (see the module docstring).

    Returns
    -------
    dict with keys:
        x : ndarray, shape (n_samples, ncomp)
            Per-sample scores (the data projected onto each component).
        rotation : ndarray, shape (n_features, ncomp)
            Loadings; non-negative on every component when ``nneg=True``.
        sdev : ndarray, shape (ncomp,)
            Standard deviation explained by each component.
        suboptimal : ndarray of bool, shape (ncomp,)
            True where the component is provably a local optimum (see the module
            docstring); a LocalOptimumWarning is issued if any is True.
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
    if random_state is not None and not isinstance(random_state, np.random.SeedSequence):
        random_state = np.random.SeedSequence(random_state)

    W = np.zeros((d, ncomp))
    Q = np.zeros((d, ncomp))
    sdev = []
    suboptimal = np.zeros(ncomp, dtype=bool)
    prev = None                      # (c, s, q, objective) of the previous component

    for cc in range(ncomp):
        best_obj = -np.inf
        best_w = None

        for r in range(nrestart):
            starts = _global_starts if random_state is None else _keyed_starts(random_state, cc, r)
            w, obj = _empca_refined(em.gram, em.sub, d, Q[:, :cc], nneg, em_tol, em_maxiter, starts)
            if obj > best_obj:
                best_obj = obj
                best_w = w.copy()

        w = best_w
        W[:, cc] = w
        # R: sd(Xp %*% w), divisor n - 1
        _, ssq = em.gram(w)
        sdev.append(np.sqrt(max(ssq - n * float(mean @ w) ** 2, 0.0) / (n - 1)))

        if prev is not None:
            # w on the previous component's problem: w' C_prev w, with C_prev = Xp' Xp
            # before the last deflation, divided by the squared norm of w orthogonal
            # to the deflation vectors before that one
            c_p, s_p, q_p, obj_p = prev
            t = float(q_p @ w)
            u = w - Q[:, :cc - 1] @ (Q[:, :cc - 1].T @ w)
            uu = float(u @ u)
            if uu > 0 and (ssq + 2 * float(c_p @ w) * t - s_p * t * t) / uu > obj_p * (1 + em_tol):
                suboptimal[cc - 1] = True

        if cc > 0:
            q = w - Q[:, :cc] @ (Q[:, :cc].T @ w)
        else:
            q = w.copy()

        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        Q[:, cc] = q
        c_p, s_p = em.deflate(q)
        prev = (c_p, s_p, q, ssq / norm ** 2 if norm > 0 else np.inf)
        mean -= float(mean @ q) * q

        if em.residual() <= _STOP_RTOL * total:
            break

    suboptimal = suboptimal[:len(sdev)]
    if suboptimal.any():
        which = ", ".join(str(k + 1) for k in np.flatnonzero(suboptimal))
        warnings.warn(f"component(s) {which}: a later component explains more of the data left at that step, "
                      "so a better non-negative solution exists; increase nrestart. This check cannot "
                      "detect every such case.", LocalOptimumWarning, stacklevel=2)

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
        "suboptimal": suboptimal,
    }
