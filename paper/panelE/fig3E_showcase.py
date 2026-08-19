"""
Panel E — visualization showcase on the FULL A549 TGF-β single-cell matrix
(Cook 2020), induction time points only (0d, 8h, 1d, 3d, 7d).
 left  : per-cell nnPCA E-vs-M scatter (standardized scores) — thresholded per-time
         KDE density fill, low-alpha points, per-time centroids with 95% bootstrap
         CI crosses (full-refit bootstrap).
 right : heatmap of the top-5 E-PC1, M-PC1 and M-PC2 loading genes across cells
         grouped by ordered time point, z-scored total-library log-CPM expression,
         with a time-point colour strip and a PC-group row annotation.
Scoring: total-library log-CPM + per-gene z-score + nsprcomp; scores standardized;
signs anchored to pseudotime.  draw_panelE() is reused by fig3_combined.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc_full  # noqa: E402

REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
DATA = REPO / "data"
RES = Path(__file__).resolve().parent / "results"
PLOTS = Path(__file__).resolve().parent / "plots"
# results/ and plots/ are gitignored, so create them on a fresh checkout
RES.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
DS = "A549_TGFB1"
TIME_ORDER = ["0d", "8h", "1d", "3d", "7d"]
CAP, B_BOOT = 130, 300
E1_COL, PC1_COL, PC2_COL = "#1f6feb", "#0f8a7e", "#c8811a"
FS_LAB, FS_TICK, FS_GENE = 11, 9.5, 8


def parse_gmt(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        x = line.rstrip("\n").split("\t")
        if len(x) >= 3:
            d[x[0]] = [g for g in x[2:] if g]
    return d


def _load():
    X, g2i, genes, lib, meta = sc_full.load(DS)
    ind = np.where(meta["Time"].isin(TIME_ORDER).values)[0]
    return X, g2i, lib, meta, ind


def _score(X, lib, cell_idx, gidx, ncomp, positive_with_pt, pt):
    r = sc_full.nnpca(X[np.ix_(cell_idx, gidx)], lib[cell_idx], ncomp)
    s = r["x"][:, 0]
    if (np.corrcoef(pt, s)[0, 1] > 0) != positive_with_pt:
        s = -s
    return sc_full.zc(s), r["rotation"]


def _prepare():
    X, g2i, lib, meta, ind = _load()
    pt = meta["Pseudotime"].values[ind]
    times = meta["Time"].astype(str).values[ind]
    Eg = [g for g in pd.read_csv(DATA / "Panchy_et_al_E_signature.csv")["GeneName"] if g in g2i]
    Mg = [g for g in pd.read_csv(DATA / "Panchy_et_al_M_signature.csv")["GeneName"] if g in g2i]
    Eidx = np.array([g2i[g] for g in Eg]); Midx = np.array([g2i[g] for g in Mg])
    sE, Erot = _score(X, lib, ind, Eidx, 1, False, pt)
    sM, Mrot = _score(X, lib, ind, Midx, 2, True, pt)
    e1 = pd.Series(np.abs(Erot[:, 0]), index=Eg).nlargest(5).index.tolist()
    pc1 = pd.Series(np.abs(Mrot[:, 0]), index=Mg).nlargest(5).index.tolist()
    pc2 = [g for g in pd.Series(np.abs(Mrot[:, 1]), index=Mg).nlargest(9).index if g not in pc1][:5]
    pdf = pd.DataFrame({"Time": times, "Escore": sE, "Mscore": sM}, index=np.arange(len(ind)))
    return X, g2i, lib, ind, pt, times, pdf, e1, pc1, pc2, Eidx, Midx


def _bootstrap_ci(X, lib, ind, pt, times, Eidx, Midx):
    cache = RES / f"em_scatter_ci_full_{DS}.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index("time")
    order = [t for t in TIME_ORDER if t in set(times)]
    clusters = [np.where(times == t)[0] for t in order]
    sizes = [len(c) for c in clusters]
    dE = {t: [] for t in order}; dM = {t: [] for t in order}
    for b in range(B_BOOT):
        rng = np.random.default_rng(11 + b)
        blocks = [c[rng.integers(0, len(c), len(c))] for c in clusters]
        idx = np.concatenate(blocks); cell = ind[idx]; ptb = pt[idx]
        sE, _ = _score(X, lib, cell, Eidx, 1, False, ptb)
        sM, _ = _score(X, lib, cell, Midx, 1, True, ptb)
        off = 0
        for t, nn in zip(order, sizes):
            sl = slice(off, off + nn); off += nn
            dE[t].append(sE[sl].mean()); dM[t].append(sM[sl].mean())
    rows = []
    for t in order:
        e, m = np.array(dE[t]), np.array(dM[t])
        rows.append({"time": t, "E_mean": e.mean(), "E_lo": np.percentile(e, 2.5), "E_hi": np.percentile(e, 97.5),
                     "M_mean": m.mean(), "M_lo": np.percentile(m, 2.5), "M_hi": np.percentile(m, 97.5)})
    out = pd.DataFrame(rows).set_index("time"); out.to_csv(cache)
    return out


def _kde_fill(ax, x, y, color, levels=7, thresh=0.12):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 5:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return
    px, py = 0.12 * (np.ptp(x) + 1e-9), 0.12 * (np.ptp(y) + 1e-9)
    xx, yy = np.mgrid[x.min() - px:x.max() + px:90j, y.min() - py:y.max() + py:90j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    lv = np.linspace(thresh * zz.max(), zz.max(), levels)
    cmap = LinearSegmentedColormap.from_list("_k", [to_rgba(color, 0.15), to_rgba(color, 0.55)])
    ax.contourf(xx, yy, zz, levels=lv, cmap=cmap, antialiased=True, zorder=1, extend="neither")


def _scatter(ax, pdf, present, cmap_t, ci):
    for t in present:
        s = pdf[pdf["Time"] == t]
        _kde_fill(ax, s["Escore"].to_numpy(), s["Mscore"].to_numpy(), cmap_t[t])
    for t in present:
        s = pdf[pdf["Time"] == t]
        ax.scatter(s["Escore"], s["Mscore"], s=7, alpha=0.16, color=cmap_t[t], edgecolors="none", zorder=2)
    handles = []
    for t in present:
        r = ci.loc[t]
        xerr = [[r.E_mean - r.E_lo], [r.E_hi - r.E_mean]]; yerr = [[r.M_mean - r.M_lo], [r.M_hi - r.M_mean]]
        ax.errorbar(r.E_mean, r.M_mean, xerr=xerr, yerr=yerr, fmt="none", ecolor="white",
                    elinewidth=5.5, capsize=8, capthick=5.5, zorder=3.5)
        ax.errorbar(r.E_mean, r.M_mean, xerr=xerr, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=2.2, capsize=6, capthick=2.2, zorder=4)
        handles.append(ax.scatter([r.E_mean], [r.M_mean], s=95, color=cmap_t[t], edgecolors="white",
                                  linewidths=2.0, zorder=5, label=t))
    ax.set_xlabel("Standardized nnPCA E score", fontsize=FS_LAB)
    ax.set_ylabel("Standardized nnPCA M score", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK); ax.grid(alpha=0.15)
    ax.legend(handles=handles, title="Time point (● mean ± 95% CI)", loc="upper right", fontsize=FS_TICK,
              title_fontsize=FS_TICK, frameon=True, facecolor="white", edgecolor="0.8", framealpha=0.9)
    ax.set_box_aspect(1)


def _heatmap(fig, spec, X, g2i, lib, ind, pdf, times, present, cmap_t, groups):
    genes = [g for _, gl, _ in groups for g in gl]
    gidx = np.array([g2i[g] for g in genes])
    cols = []
    for t in present:
        sel = np.where(pdf["Time"].values == t)[0]
        sub = pdf.iloc[sel].sort_values("Mscore")
        if len(sub) > CAP:
            sub = sub.iloc[np.linspace(0, len(sub) - 1, CAP).astype(int)]
        cols.extend(sub.index.tolist())
    cols = np.array(cols)
    times_ord = times[cols]
    Z = sc_full.prep(X[np.ix_(ind[cols], gidx)], lib[ind[cols]]).T      # genes × cells

    hgs = spec.subgridspec(2, 4, height_ratios=[0.05, 1.0], width_ratios=[0.05, 1.0, 0.14, 0.035],
                           hspace=0.02, wspace=0.02)
    ax_ts = fig.add_subplot(hgs[0, 1]); ax_pc = fig.add_subplot(hgs[1, 0])
    ax_hm = fig.add_subplot(hgs[1, 1]); ax_cb = fig.add_subplot(hgs[1, 3])

    ax_ts.imshow(np.array([[mcolors.to_rgb(cmap_t[t]) for t in times_ord]]), aspect="auto", interpolation="none")
    ax_ts.set_xticks([]); ax_ts.set_yticks([])
    for sp in ax_ts.spines.values():
        sp.set_visible(False)
    b = 0
    for t in present:
        nn = int((times_ord == t).sum())
        ax_ts.text(b + nn / 2, -0.8, t, ha="center", va="bottom", fontsize=FS_TICK, color="#333")
        b += nn
        if b < len(times_ord):
            ax_hm.axvline(b - 0.5, color="white", lw=1.1)

    strip = []
    for _, gl, col in groups:
        strip += [[mcolors.to_rgb(col)]] * len(gl)
    ax_pc.imshow(np.array(strip), aspect="auto", interpolation="none")
    ax_pc.set_xticks([]); ax_pc.set_yticks([])
    off = 0
    for name, gl, col in groups:
        ax_pc.text(-1.6, off + (len(gl) - 1) / 2, name, rotation=90, va="center", ha="center",
                   fontsize=FS_TICK, color=col, fontweight="bold")
        off += len(gl)
    for sp in ax_pc.spines.values():
        sp.set_visible(False)

    vmax = float(np.nanpercentile(np.abs(Z), 98)) or 1.0
    im = ax_hm.imshow(Z, aspect="auto", cmap="RdBu_r",
                      norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), interpolation="none")
    off = 0
    for name, gl, col in groups[:-1]:
        off += len(gl); ax_hm.axhline(off - 0.5, color="black", lw=1.2)
    ax_hm.set_xticks([]); ax_hm.set_yticks(range(len(genes)))
    ax_hm.set_yticklabels(genes, fontsize=FS_GENE)
    ax_hm.yaxis.tick_right(); ax_hm.tick_params(right=False)
    ax_hm.set_xlabel(f"Cells grouped by time point  (n={len(cols)})", fontsize=FS_LAB)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=ax_cb); cb.set_label("Expression (z-score)", fontsize=FS_TICK)
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    return ax_ts


def draw_panelE(fig, spec_scatter, spec_heatmap):
    X, g2i, lib, ind, pt, times, pdf, e1, pc1, pc2, Eidx, Midx = _prepare()
    ci = _bootstrap_ci(X, lib, ind, pt, times, Eidx, Midx)
    present = [t for t in TIME_ORDER if t in set(times)]
    cmap_t = {t: plt.cm.viridis(v) for t, v in zip(present, np.linspace(0.05, 0.95, len(present)))}
    groups = [("E PC1", e1, E1_COL), ("M PC1", pc1, PC1_COL), ("M PC2", pc2, PC2_COL)]
    axL = fig.add_subplot(spec_scatter)
    _scatter(axL, pdf, present, cmap_t, ci)
    axtop = _heatmap(fig, spec_heatmap, X, g2i, lib, ind, pdf, times, present, cmap_t, groups)
    return axL, axtop


def main():
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.20,
                          left=0.06, right=0.955, top=0.92, bottom=0.12)
    draw_panelE(fig, gs[0, 0], gs[0, 1])
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"fig3E_showcase.{ext}", dpi=150, bbox_inches="tight")
    print("wrote fig3E_showcase")


if __name__ == "__main__":
    main()
