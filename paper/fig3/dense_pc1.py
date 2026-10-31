"""nnPCA vs dense PC1 (PLAGE-style) scores across the C2 sets of the Fig 3D screen.

Both scores use the same input as c2_screen.py: library-size log-CPM, z-scored per gene,
centred. The dense score is each set's first principal component (top eigenvector of the gene
covariance), the PLAGE score up to its sign; the sign is chosen so that the score correlates
positively with the set's mean z-score. The nnPCA score is the first non-negative component
with 50 restarts and the screen's seeds, so its correlation with M equals the screen's point
estimate. Per set:

  neg_share         share of the dense loading's squared norm on genes with negative loadings
  eig_ratio         lambda2 / lambda1 of the set's covariance (near 1: dense PC1 is ill-defined)
  var_kept          variance of the nnPCA score / lambda1 (what non-negativity costs, <= 1)
  support           share of the set's genes with a nonzero nnPCA loading
  cos_load          cosine between the two loading vectors
  r_scores          Spearman r between the two scores
  rM_nn, rM_dense   Spearman r with the M score, M scored by the same method

Writes results/dense_pc1.csv.

Usage:
    python dense_pc1.py              # about 5 min
    python dense_pc1.py --limit 200  # quick check
"""
from __future__ import annotations

import argparse

import c2_screen  # first: sets one BLAS thread, as in the screen
import numpy as np
import pandas as pd
import resample as rs

_ranks, _sets, TOP = c2_screen._ranks, c2_screen._sets, c2_screen.TOP


def _r(a, b):
    """Correlation of two centred rank vectors (Spearman r)."""
    d = np.sqrt((a @ a) * (b @ b))
    return 0.0 if d == 0 else float(a @ b / d)


def dense_pc1(Zs: np.ndarray):
    """First principal component of the columns of Zs: loading, score, lambda1, lambda2."""
    Zc = Zs - Zs.mean(axis=0)
    lam, V = np.linalg.eigh(Zc.T @ Zc / (len(Zc) - 1))
    w = V[:, -1]
    x = Zc @ w
    if np.corrcoef(x, Zc.mean(axis=1))[0, 1] < 0:
        w, x = -w, -x
    return w, x, lam[-1], lam[-2] if len(lam) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=rs.SEED)
    ap.add_argument("--limit", type=int, default=None, help="only the first N sets (quick check)")
    a = ap.parse_args()

    W = _sets(a.limit)
    Z = np.empty(W["X"].shape, np.float32)
    for j in range(0, Z.shape[1], 1024):
        Z[:, j:j + 1024] = rs.zscores(W["X"][:, j:j + 1024], W["lib"])

    def start(k):                                    # the screen's point-estimate seeds
        return np.random.SeedSequence([a.seed, 0, k])

    xM_nn = _ranks(rs.fit(Z[:, W["colM"]], 1, start(0), rs.REF_NRESTART)["x"][:, 0])
    xM_dense = _ranks(dense_pc1(Z[:, W["colM"]].astype(float))[1])
    rows = []
    for k, c in enumerate(W["col"]):
        w, x, lam1, lam2 = dense_pc1(Z[:, c].astype(float))
        f = rs.fit(Z[:, c], 1, start(k + 1), rs.REF_NRESTART)
        v, y = f["rotation"][:, 0], f["x"][:, 0]
        ry, rx = _ranks(y), _ranks(x)
        rows.append({"gene_set": W["names"][k], "n_genes": len(c),
                     "neg_share": float((w[w < 0] ** 2).sum()), "eig_ratio": float(lam2 / lam1),
                     "var_kept": float(y.var(ddof=1) / lam1), "support": float((v > 0).mean()),
                     "cos_load": float(v @ w / np.linalg.norm(v)), "r_scores": _r(ry, rx),
                     "rM_nn": _r(ry, xM_nn), "rM_dense": _r(rx, xM_dense)})
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(W['col'])} sets", flush=True)
    d = pd.DataFrame(rows)
    d["rank_nn"] = d.rM_nn.rank(ascending=False, method="first").astype(int)
    d["rank_dense"] = d.rM_dense.rank(ascending=False, method="first").astype(int)
    d.sort_values("r_scores").to_csv(rs.RESULTS / "dense_pc1.csv", index=False)

    point = rs.RESULTS / "c2_screen" / f"{len(d)}sets_seed{a.seed}" / "point.csv"
    if point.exists():
        p = pd.read_csv(point).set_index("gene_set").r_point.reindex(d.gene_set).to_numpy()
        print(f"check: r with M vs the screen's point estimates, max |diff| {np.abs(p - d.rM_nn).max():.2g}")
    q = d.r_scores
    print(f"{len(d)} sets; M scored both ways: r = {_r(xM_nn, xM_dense):.3f}")
    print(f"r between the scores: median {q.median():.3f}; > 0.95 in {(q > .95).mean():.0%} of sets, "
          f"< 0.8 in {(q < .8).mean():.0%}, < 0 in {(q < 0).sum()}")
    print(f"Spearman with r_scores: neg_share {d.neg_share.corr(q, method='spearman'):.2f}, "
          f"eig_ratio {d.eig_ratio.corr(q, method='spearman'):.2f}, n_genes {d.n_genes.corr(q, method='spearman'):.2f}")
    print(f"var_kept: median {d.var_kept.median():.3f}, min {d.var_kept.min():.3f}; "
          f"support: median {d.support.median():.2f}")

    def ends(m):
        return set(d.nlargest(TOP, m).gene_set) | set(d.nsmallest(TOP, m).gene_set)

    flip = (np.sign(d.rM_nn) != np.sign(d.rM_dense)) & (np.maximum(d.rM_nn.abs(), d.rM_dense.abs()) > 0.3)
    print(f"top and bottom {TOP} by r with M: {len(ends('rM_nn') & ends('rM_dense'))} of {2 * TOP} shared; "
          f"r with M changes sign (either |r| > 0.3) in {flip.sum()} sets")
    with pd.option_context("display.width", 200, "display.max_colwidth", 44):
        print(d.nsmallest(15, "r_scores").drop(columns=["rank_nn", "rank_dense"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
