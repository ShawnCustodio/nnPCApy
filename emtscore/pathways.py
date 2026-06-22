"""
Section 3.3: Pathway correlation analysis
=========================================
Maps to R EMTscore: filter_gmt_by_reference (with overlap_max=0.3) +
Execute_nnPCA_per_pathway + cor() loop.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from utility.data_paths import resolve_data_file
from .nsprcomp import nsprcomp


# ─── helpers ─────────────────────────────────────────────────────────────────

def _parse_gmt_3_3(path: Path) -> dict[str, list[str]]:
    """Load a GMT into {set_name: [gene, gene, ...]} (gene symbols upper-cased)."""
    gmt: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gmt[parts[0]] = [g.upper() for g in parts[2:] if g]
    return gmt


def _filter_pathways_by_overlap(
    gmt: dict[str, list[str]],
    reference_genes: set[str],
    max_overlap_frac: float = 0.30,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """Drop pathways whose genes overlap > max_overlap_frac with reference.

    overlap = |P ∩ reference| / |P|   (relative to pathway size)

    Following the EMTscore convention: if more than 30% of a pathway's
    genes are also in the EMT (M) gene set, the pathway is too redundant
    with the EMT signature itself to carry useful biological information
    about EMT-correlated processes — so discard it.
    """
    kept: dict[str, list[str]] = {}
    dropped_high_overlap: list[tuple[str, float]] = []

    for name, genes in gmt.items():
        gset = {g.upper() for g in genes}
        if not gset:
            continue
        overlap = len(gset & reference_genes) / len(gset)
        if overlap > max_overlap_frac:
            dropped_high_overlap.append((name, overlap))
        else:
            kept[name] = genes

    if verbose:
        print(f"[overlap filter] kept {len(kept)} / {len(gmt)} pathways "
              f"(dropped {len(dropped_high_overlap)} with > "
              f"{max_overlap_frac * 100:.0f}% overlap with EMT M signature)")
        if dropped_high_overlap:
            top5 = sorted(dropped_high_overlap, key=lambda kv: -kv[1])[:5]
            for name, ov in top5:
                print(f"   dropped: {name}  ({ov*100:.1f}% in M)")
    return kept


def _score_one_pathway(name: str, genes: list[str],
                        expr_genes_x_samples: pd.DataFrame) -> pd.Series | None:
    """Top-PC1 nnPCA score per sample for one pathway. Returns None if too few genes."""
    shared = [g for g in genes if g in expr_genes_x_samples.index]
    if len(shared) < 3:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = nsprcomp(expr_genes_x_samples.loc[shared].T.values,
                           ncomp=1, nneg=True, center=True, scale_=False)
        return pd.Series(res["x"][:, 0],
                         index=expr_genes_x_samples.columns, name=name)
    except Exception:
        return None


def _score_all_pathways(
    expr_genes_x_samples: pd.DataFrame,
    gmt: dict[str, list[str]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every pathway → DataFrame indexed by samples, columns = pathways."""
    results: dict[str, pd.Series] = {}
    for i, (name, genes) in enumerate(gmt.items(), 1):
        s = _score_one_pathway(name, genes, expr_genes_x_samples)
        if s is not None:
            results[name] = s
        if verbose and i % 500 == 0:
            print(f"  scored {i}/{len(gmt)} pathways...")
    mat = pd.DataFrame(results)
    if verbose:
        print(f"[pathway scores] {mat.shape[0]} samples × {mat.shape[1]} pathways")
    return mat


def _correlate_against_emt(
    pathway_scores: pd.DataFrame,
    emt_score: pd.Series,
) -> pd.DataFrame:
    """Pearson correlate every pathway column against the single EMT score."""
    shared = pathway_scores.index.intersection(emt_score.index)
    if len(shared) == 0:
        raise ValueError("No common samples between pathway scores and EMT score.")

    P = pathway_scores.loc[shared]
    y = emt_score.loc[shared].values

    rows = []
    for col in P.columns:
        x = P[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() <= 2:
            continue
        try:
            r, p = pearsonr(x[valid], y[valid])
        except Exception:
            continue
        rows.append({
            "Pathway":     col,
            "Correlation": float(r),
            "P_value":     float(p),
        })

    if not rows:
        raise ValueError("No valid correlations computed.")

    df = pd.DataFrame(rows)
    # Sort by signed correlation, descending → easy positive top, ascending tail = negative top
    df = df.sort_values("Correlation", ascending=False).reset_index(drop=True)
    return df


# ─── public API ──────────────────────────────────────────────────────────────

def run_pathway_correlation_v2(
    verbose: bool = True,
    random_state: int = 42,
    max_overlap_frac: float = 0.30,
) -> pd.DataFrame:
    """
    3.3 — single-EMT-score, overlap-filtered pathway correlation.
    """
    np.random.seed(random_state)

    GENEEXP_CSV = resolve_data_file("geneExp.csv")
    M_SIG_CSV   = resolve_data_file("Panchy_et_al_M_signature.csv")
    C2_GMT      = resolve_data_file("c2.all.v2025.1.Hs.symbols.gmt")

    # ── load expression (genes × samples), upper-case gene symbols ──
    _raw = pd.read_csv(GENEEXP_CSV, index_col=0)
    expr = _raw if _raw.shape[0] > _raw.shape[1] else _raw.T
    expr.index = expr.index.str.upper()
    if verbose:
        print(f"[geneExp] {expr.shape[0]} genes × {expr.shape[1]} samples")

    # ── M signature genes  ──
    M_genes = pd.read_csv(M_SIG_CSV)["GeneName"].dropna().str.upper().tolist()
    M_set = set(M_genes)
    if verbose:
        print(f"[M signature] {len(M_set)} unique genes")

    # ── Step 1: single EMT score = top-PC1 nnPCA on M signature ──
    if verbose:
        print("\n[Step 1] Computing single EMT score (M-signature top PC1)...")
    M_in_expr = [g for g in M_genes if g in expr.index]
    if len(M_in_expr) < 3:
        raise ValueError(
            f"Only {len(M_in_expr)} of {len(M_genes)} M-signature genes "
            f"are present in geneExp."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = nsprcomp(expr.loc[M_in_expr].T.values,
                       ncomp=1, nneg=True, center=True, scale_=False)
    emt_score = pd.Series(res["x"][:, 0], index=expr.columns, name="EMT_M_PC1")
    if verbose:
        print(f"[EMT score] {emt_score.shape[0]} samples (range: "
              f"{emt_score.min():.3f} to {emt_score.max():.3f})")

    # ── Step 2: load + filter C2 ──
    if verbose:
        print("\n[Step 2] Filtering C2 pathways by M-overlap...")
    c2 = _parse_gmt_3_3(C2_GMT)
    c2_filt = _filter_pathways_by_overlap(c2, M_set, max_overlap_frac, verbose)

    # ── Step 3: nnPCA-score every kept pathway ──
    if verbose:
        print("\n[Step 3] Scoring filtered C2 pathways...")
    pathway_scores = _score_all_pathways(expr, c2_filt, verbose=verbose)

    # ── Step 4: Pearson correlate against the single EMT score ──
    if verbose:
        print("\n[Step 4] Correlating pathway scores against EMT M PC1...")
    result = _correlate_against_emt(pathway_scores, emt_score)

    # Back-compat columns so the existing notebook plot helper keeps working
    result["Pathway_in_score_mat1"] = result["Pathway"]
    result["Pathway_in_score_mat2"] = "EMT_M_PC1"

    if verbose:
        print(f"\n[result] {len(result)} pathways with valid correlations.")
        print(f"[top +]  {result.iloc[0]['Pathway']}  r={result.iloc[0]['Correlation']:+.3f}")
        print(f"[top -]  {result.iloc[-1]['Pathway']}  r={result.iloc[-1]['Correlation']:+.3f}")
    return result


# ─── 3.3.1 / 3.3.2  Plot top correlated pathways (plotly bar) ──────────────

def plot_top_pathways(result: pd.DataFrame, n: int = 10, mode: str = "positive",
                      title: str | None = None, color: str = "#84C7B9"):
    import plotly.graph_objects as go

    if mode == "positive":
        df = result.head(n).copy()
        if title is None:
            title = f"Top {n} Positive Correlated Pathways"
    elif mode == "negative":
        sig = result[result["P_value"] < 0.05]
        df = sig.sort_values("Correlation", ascending=True).head(n).copy()
        if title is None:
            title = f"Top {n} Negative Correlated Pathways"
    else:
        raise ValueError("mode must be 'positive' or 'negative'")

    df = df.sort_values("Correlation", ascending=True).reset_index(drop=True)

    fig = go.Figure(go.Bar(
        x=df["Correlation"], y=df["Pathway_in_score_mat1"],
        orientation="h", marker_color=color,
        text=df["Pathway_in_score_mat1"], textposition="inside",
        insidetextanchor=("start" if mode == "positive" else "end"),
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "vs %{customdata[0]}<br>"
            "r = %{x:.3f}<br>"
            "p = %{customdata[1]:.2e}<extra></extra>"
        ),
        customdata=df[["Pathway_in_score_mat2", "P_value"]].values,
    ))
    fig.update_layout(
        title=dict(text=title, font_size=16),
        xaxis_title="Correlation", yaxis_title="",
        yaxis=dict(showticklabels=False),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(l=20, r=30, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor="#eee",
                   zeroline=True, zerolinecolor="#ccc"),
    )
    return fig


__all__ = [
    "run_pathway_correlation_v2",
    "plot_top_pathways",
]
