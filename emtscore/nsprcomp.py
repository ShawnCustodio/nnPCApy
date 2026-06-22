import numpy as np
import pandas as pd

def normv(w):
    return np.linalg.norm(w)


def empca(Xp, Q, nneg, em_tol, em_maxiter):
    d = Xp.shape[1]

    # R-like initialization
    w = np.random.randn(d)
    if nneg:
        w = np.abs(w)

    w = w / (normv(w) + 1e-12)

    obj_old = -np.inf

    for _ in range(em_maxiter):
        z = Xp @ w
        obj = float(z.T @ z)

        # convergence check
        if obj != 0 and abs(obj - obj_old) / (abs(obj_old) + 1e-12) < em_tol:
            break
        obj_old = obj

        # update step
        denom = obj if obj != 0 else 1e-12
        w_star = Xp.T @ z / denom

        # non-negativity constraint
        if nneg:
            w_star[w_star < 0] = 0

       
        w = w_star

        # orthogonalization against previous components
        if Q.shape[1] > 0:
            w = w - Q @ (Q.T @ w)

        norm = normv(w)
        if norm < 1e-12:
            break

        w = w / norm

    return w, obj


def nsprcomp(
    x,
    ncomp=1,
    center=True,
    scale_=False,
    nneg=True,
    nrestart=20,
    em_tol=1e-4,
    em_maxiter=200
):
    """
    Returns PCA-like components with optional non-negativity constraint.
    """

    X = x.astype(np.float64)

    # ----------------------------
    # Centering 
    # ----------------------------
    if center:
        cen = X.mean(axis=0)
        X = X - cen
    else:
        cen = np.zeros(X.shape[1])

    # ----------------------------
    # Scaling 
    # ----------------------------
    if scale_:
        sc = X.std(axis=0)
        sc[sc == 0] = 1
        X = X / sc
    else:
        sc = np.ones(X.shape[1])

    Xp = X.copy()

    n, d = X.shape
    W = np.zeros((d, ncomp))
    Q = np.zeros((d, ncomp))

    sdev = []

    for cc in range(ncomp):
        best_obj = -np.inf
        best_w = None

        for _ in range(nrestart):
            w, obj = empca(
                Xp,
                Q[:, :cc],
                nneg,
                em_tol,
                em_maxiter
            )

            # keep best restart
            if obj > best_obj:
                best_obj = obj
                best_w = w.copy()

        w = best_w
        W[:, cc] = w

        sdev.append(np.std(Xp @ w))

        # Gram-Schmidt orthogonalization
        if cc > 0:
            q = w - Q[:, :cc] @ (Q[:, :cc].T @ w)
        else:
            q = w.copy()

        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        Q[:, cc] = q

        # deflation 
        Xp = Xp - np.outer(Xp @ q, q)

        # early stop if matrix collapses
        if np.all(np.abs(Xp) < 1e-14):
            break

    return {
        "sdev": np.array(sdev),
        "rotation": W,
        "center": cen,
        "scale": sc,
        "x": X @ W   # sample scores
    }
    
    
def compute_M1_M2_scores(
    geneExp: pd.DataFrame,
    genes: list,
    perturbation: float = 1e-4,
    n_components: int = 2,
) -> pd.DataFrame:
    """
    Compute M1/M2 EMT scores using nsprcomp.
    """

    available = [g for g in genes if g in geneExp.columns]

    if len(available) == 0:
        return pd.DataFrame(
            np.zeros((geneExp.shape[0], n_components)),
            index=geneExp.index,
            columns=[f"M{i+1}" for i in range(n_components)]
        )

    X = geneExp[available].values.astype(np.float64)

    # small noise to avoid degeneracy
    np.random.seed(42)
    X = X + np.random.randn(*X.shape) * perturbation

    result = nsprcomp(X, ncomp=n_components, nneg=True, center=True, scale_=False)

    scores = result["x"][:, :n_components]

    return pd.DataFrame(
        scores,
        index=geneExp.index,
        columns=[f"M{i+1}" for i in range(n_components)]
    )