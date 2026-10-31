"""Screen of the C2 collection against the M signature, with resampling (Fig 3D).

Exploratory: every C2 set with at least 15 genes in the data and at most 30% of its genes
in M is scored with nnPCA (one component), and its per-cell score is Spearman-correlated
with the M score. The point estimates use the full data and 50 restarts. Each bootstrap
draw resamples cells within time point x culture, refits M and every set on the same
cells (20 restarts) and recomputes every correlation, so each draw also ranks all sets.
Per set this gives a percentile interval for r, an interval for its rank, and how often it
is among the 15 most positive or the 15 most negative sets (inclusion frequency). These are
ordinary bootstrap summaries for sets chosen by their extreme correlations: not
selection-adjusted intervals and not tests.

Draws run in --workers processes and are saved one file per draw in
results/c2_screen/<sets>sets_seed<seed>/, so an interrupted run continues where it stopped.
Writes results/c2_screen.csv.

Usage:
    python c2_screen.py --B 300 --workers 2    # ~2-4 min per draw per worker, ~0.4 GB each
    python c2_screen.py --B 4 --workers 2 --limit 200   # quick check on 200 sets
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # one BLAS thread per worker process

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import resample as rs  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

MIN_GENES, OVERLAP_MAX, TOP = 15, 0.30, 15
C2 = "c2.all.v2025.1.Hs.symbols.gmt"
M_SET = "Panchy_et_al_M_signature"

_W: dict = {}                          # per-process data, filled by _setup


def _sets(limit=None):
    X, g2i, _genes, lib, meta = rs.load()
    M = [g for g in rs.parse_gmt(rs.DATA / "EM_signature.gmt")[M_SET] if g in g2i]
    Mset = set(M)
    sets = {}
    for name, gg in rs.parse_gmt(rs.DATA / C2).items():
        gp = list(dict.fromkeys(g for g in gg if g in g2i))
        if len(gp) >= MIN_GENES and len(set(gp) & Mset) / len(gp) <= OVERLAP_MAX:
            sets[name] = np.array([g2i[g] for g in gp])
    if limit:
        sets = dict(list(sets.items())[:limit])
    Midx = np.array([g2i[g] for g in M])
    union = np.unique(np.concatenate([Midx, *sets.values()]))
    X = np.ascontiguousarray(X[:, union])            # keep only the screened genes
    rs.sc_full._CACHE.clear()
    return {"X": X, "lib": lib, "groups": rs.strata(meta["Time"].to_numpy(), rs.cultures(meta)),
            "names": list(sets), "sizes": [len(v) for v in sets.values()], "union": union,
            "colM": np.searchsorted(union, Midx), "col": [np.searchsorted(union, v) for v in sets.values()]}


def _setup(limit):
    _W.update(_sets(limit))


def _zscored(rows, block=1024):
    """rs.zscores of the resampled cells, in column blocks and as float32 to bound memory."""
    Xr, lib = _W["X"][rows], _W["lib"][rows]
    Z = np.empty(Xr.shape, np.float32)
    for j in range(0, Xr.shape[1], block):
        Z[:, j:j + block] = rs.zscores(Xr[:, j:j + block], lib)
    return Z


def _ranks(x):
    return rankdata(x) - (len(x) + 1) / 2


def _correlations(rows, start, nrestart):
    """Spearman r of every set's score with the M score, all refitted on these cells."""
    Z = _zscored(rows)
    rM = _ranks(rs.fit(Z[:, _W["colM"]], 1, start(0), nrestart)["x"][:, 0])
    out = np.empty(len(_W["col"]))
    for k, c in enumerate(_W["col"]):
        r = _ranks(rs.fit(Z[:, c], 1, start(k + 1), nrestart)["x"][:, 0])
        d = np.sqrt((r @ r) * (rM @ rM))
        out[k] = 0.0 if d == 0 else r @ rM / d
    return out


def _draw(b, seed, out):
    n = _W["X"].shape[0]
    rows = rs.resample_rows(n, np.random.default_rng([seed, 2, b]), _W["groups"])
    r = _correlations(rows, lambda k: np.random.SeedSequence([seed, 3, b, k]), rs.NRESTART)
    tmp = out / f"b{b:05d}.tmp.npy"
    np.save(tmp, r.astype(np.float32))
    tmp.replace(out / f"b{b:05d}.npy")       # a killed run never leaves a partial draw file
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=rs.SEED)
    ap.add_argument("--limit", type=int, default=None, help="only the first N sets (quick check)")
    a = ap.parse_args()

    _setup(a.limit)
    names, sizes, n = _W["names"], _W["sizes"], _W["X"].shape[0]
    K = len(names)
    out_dir = rs.RESULTS / "c2_screen" / f"{K}sets_seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    point_csv = out_dir / "point.csv"
    if point_csv.exists() and len(pd.read_csv(point_csv)) == K:
        r_point = pd.read_csv(point_csv)["r_point"].to_numpy()
    else:
        t0 = time.perf_counter()
        r_point = _correlations(np.arange(n), lambda k: np.random.SeedSequence([a.seed, 0, k]), rs.REF_NRESTART)
        pd.DataFrame({"gene_set": names, "r_point": r_point}).to_csv(point_csv.with_suffix(".tmp"), index=False)
        point_csv.with_suffix(".tmp").replace(point_csv)
        print(f"point screen: {K} sets in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"{K} sets, {len(_W['union'])} genes, {len(_W['groups'])} time x culture strata; B={a.B}, "
          f"{a.workers} workers, nrestart={rs.NRESTART}", flush=True)
    _W.clear()                                       # the workers load their own copy

    todo = [b for b in range(a.B) if not (out_dir / f"b{b:05d}.npy").exists()]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=_setup, initargs=(a.limit,)) as ex:
        futures = [ex.submit(_draw, b, a.seed, out_dir) for b in todo]
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)} draws ({el / 60:.0f} min, {el / i:.0f} s/draw overall)", flush=True)

    R = np.stack([np.load(out_dir / f"b{b:05d}.npy") for b in range(a.B)]).astype(float)   # B x K
    rank_pos = (-R).argsort(1).argsort(1) + 1          # 1 = most positive in that draw
    rank_neg = R.argsort(1).argsort(1) + 1             # 1 = most negative
    lo, hi = np.percentile(R, [2.5, 97.5], axis=0)
    point_rank = (-r_point).argsort().argsort() + 1
    out = pd.DataFrame({
        "gene_set": names, "n_genes": sizes, "r_point": r_point, "rank_point": point_rank,
        "r_median": np.median(R, 0), "r_lo": lo, "r_hi": hi,
        "rank_lo": np.percentile(rank_pos, 2.5, axis=0), "rank_hi": np.percentile(rank_pos, 97.5, axis=0),
        f"top{TOP}_pos_freq": (rank_pos <= TOP).mean(0), f"top{TOP}_neg_freq": (rank_neg <= TOP).mean(0),
        "draws": a.B})
    out.sort_values("r_point", ascending=False).to_csv(rs.RESULTS / "c2_screen.csv", index=False)
    shown = pd.concat([out.nlargest(TOP, "r_point"), out.nsmallest(TOP, "r_point")])
    with pd.option_context("display.width", 200, "display.max_columns", 12, "display.max_colwidth", 44):
        print(shown[["gene_set", "r_point", "r_lo", "r_hi", "rank_lo", "rank_hi",
                     f"top{TOP}_pos_freq", f"top{TOP}_neg_freq"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
