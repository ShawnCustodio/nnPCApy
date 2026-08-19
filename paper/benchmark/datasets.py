"""
Dataset prep for the nnPCA R-vs-Python benchmark.

Each loader returns the *exact* normalised numeric matrix that gets handed
to nsprcomp:

    rows    = samples / cells
    columns = signature genes that exist in the dataset

Both the Python and R benchmark scripts read the same intermediate
``_inputs/*.npy`` files this script writes, so the two sides start from
byte-identical inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os

# Data lookup order:
#   1. NNPCAPY_DATA_DIR env var (explicit override)
#   2. ./data/ bundled in this repo (default)
#   3. ../EMTScorePy/data/ (legacy, for sibling-checkout setups)
_HERE = Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / "pyproject.toml").exists()), _HERE.parents[1])
_CANDIDATES = [
    Path(os.environ["NNPCAPY_DATA_DIR"]) if "NNPCAPY_DATA_DIR" in os.environ else None,
    _ROOT / "data",                             # nnPCApy/data
    _ROOT.parent / "EMTScorePy" / "data",       # sibling-checkout legacy
]
DATA_DIR = next((p for p in _CANDIDATES if p and p.is_dir()), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find benchmark data. Expected nnPCApy/data/ to contain "
        "geneExp.csv, Panchy_et_al_{E,M}_signature.csv, filtered.c2.gmt, "
        "and cook2020/A549_*_em_expr.csv. Override with NNPCAPY_DATA_DIR."
    )
OUT_DIR = _HERE.parent / "_inputs"
OUT_DIR.mkdir(exist_ok=True)


def _lognorm(expr: np.ndarray, scale: float = 10_000.0) -> np.ndarray:
    # Bulk microarray data can contain NaN or negative values; clip so log1p is safe.
    expr = np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0)
    expr = np.maximum(expr, 0.0)
    lib = expr.sum(axis=1, keepdims=True)
    lib = np.where(lib == 0, 1, lib)
    return np.log1p(expr / lib * scale)


def _scale_genes(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0); sd[sd == 0] = 1
    return (X - mu) / sd


def load_panchy(letter: str) -> list[str]:
    fname = f"Panchy_et_al_{letter}_signature.csv"
    return pd.read_csv(DATA_DIR / fname)["GeneName"].astype(str).tolist()


def load_c2() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in (DATA_DIR / "filtered.c2.gmt").read_text().splitlines():
        parts = line.rstrip().split("\t")
        if len(parts) < 3:
            continue
        out[parts[0]] = parts[2:]
    return out


def load_bulk() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(DATA_DIR / "geneExp.csv", index_col=0)
    return df.T, "bulk_geneExp"


def load_sc(name: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(DATA_DIR / "cook2020" / f"{name}_em_expr.csv", index_col=0)
    meta = ["Pseudotime", "Treatment", "Time", "Cluster"]
    expr_cols = [c for c in df.columns if c not in meta]
    return df[expr_cols], f"sc_{name}"


def build_matrix(expr: pd.DataFrame, gene_set: list[str]) -> np.ndarray:
    keep = [g for g in gene_set if g in expr.columns]
    if len(keep) < 3:
        return np.empty((expr.shape[0], 0))
    sub = expr[keep].to_numpy(dtype=np.float64)
    return _scale_genes(_lognorm(sub))


DATASETS = [
    ("bulk",        load_bulk),
    ("A549_TGFB1",  lambda: load_sc("A549_TGFB1")),
    ("A549_EGF",    lambda: load_sc("A549_EGF")),
    ("A549_TNF",    lambda: load_sc("A549_TNF")),
]


def prepare_all(verbose: bool = True) -> list[dict]:
    E_genes = load_panchy("E")
    M_genes = load_panchy("M")
    c2      = load_c2()
    manifest: list[dict] = []

    for ds_name, loader in DATASETS:
        expr, _label = loader()
        if verbose:
            print(f"[{ds_name}] expr matrix: {expr.shape[0]} rows x {expr.shape[1]} genes")

        for sig_label, sig_genes in [("Panchy_E", E_genes), ("Panchy_M", M_genes)]:
            X = build_matrix(expr, sig_genes)
            if X.shape[1] < 3:
                if verbose:
                    print(f"  [skip] {sig_label}: only {X.shape[1]} genes present")
                continue
            tag = f"{ds_name}__{sig_label}"
            np.save(OUT_DIR / f"{tag}.npy", X)
            manifest.append({
                "tag": tag, "dataset": ds_name, "gene_set": sig_label,
                "mode": "single", "n_rows": X.shape[0], "n_cols": X.shape[1],
            })

        for pw_name, pw_genes in c2.items():
            X = build_matrix(expr, pw_genes)
            if X.shape[1] < 3:
                continue
            tag = f"{ds_name}__c2__{pw_name}"
            np.save(OUT_DIR / f"{tag}.npy", X)
            manifest.append({
                "tag": tag, "dataset": ds_name, "gene_set": pw_name,
                "mode": "multi", "n_rows": X.shape[0], "n_cols": X.shape[1],
            })

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    if verbose:
        n_single = sum(1 for r in manifest if r["mode"] == "single")
        n_multi  = sum(1 for r in manifest if r["mode"] == "multi")
        msg = "Manifest: " + str(len(manifest)) + " matrices ("
        msg += str(n_single) + " single-gene-set, " + str(n_multi) + " multi-gene-set)"
        print()
        print(msg)
        print("Saved to: " + str(OUT_DIR))
    return manifest


if __name__ == "__main__":
    sys.exit(0 if prepare_all() else 1)
