"""Build the input matrices for the v2 benchmark suites.

  realdata  the 47 matrices of the committed v1 benchmark (Panchy E/M and C2 pathways on
            the bulk panel and the three Cook A549 datasets), built by ../datasets.py into
            ../_inputs/. Reused unchanged so v1 and v2 numbers refer to the same inputs.
  dsweep    synthetic matrices with planted non-negative sparse structure
            (../synth/gen_synth.make_matrix), fixed N and increasing feature count D,
            to show where the covariance formulation stops paying off.
  pbmc3k    10x PBMC 3k, all detected genes (2,638 cells x 13,714 genes, sparse);
            downloaded from 10x Genomics on first use (about 7 MB).

Usage:
    python make_inputs.py                # realdata and dsweep
    python make_inputs.py realdata       # one suite
    python make_inputs.py pbmc3k
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import BENCH, INPUTS

DSWEEP_N = 5000
DSWEEP_D = [100, 200, 500, 1000, 2000, 5000]
DSWEEP_SEED = 20261029


def manifest_path(suite: str):
    return {"realdata": BENCH / "_inputs" / "manifest.csv",
            "dsweep": INPUTS / "dsweep" / "manifest.csv",
            "pbmc3k": INPUTS / "pbmc3k" / "manifest.csv"}[suite]


def input_path(suite: str, tag: str):
    if suite == "pbmc3k":
        return INPUTS / "pbmc3k" / f"{tag}.npz"          # sparse; R reads the .mtx beside it
    return {"realdata": BENCH / "_inputs", "dsweep": INPUTS / "dsweep"}[suite] / f"{tag}.npy"


def make_realdata() -> None:
    if manifest_path("realdata").exists():
        print(f"[realdata] reusing {manifest_path('realdata')}")
    else:
        sys.path.insert(0, str(BENCH))
        from datasets import prepare_all
        prepare_all()
    # Caches built before datasets.py saved in C order hold column-major files, which
    # RcppCNPy::npyLoad reads as a scrambled matrix. Rewrite them; values are unchanged.
    fixed = 0
    for tag in pd.read_csv(manifest_path("realdata"))["tag"]:
        p = input_path("realdata", tag)
        X = np.load(p)
        if not X.flags["C_CONTIGUOUS"]:
            np.save(p, np.ascontiguousarray(X))
            fixed += 1
    if fixed:
        print(f"[realdata] rewrote {fixed} column-major inputs in C order")


def make_dsweep() -> None:
    sys.path.insert(0, str(BENCH / "synth"))
    from gen_synth import make_matrix
    out = INPUTS / "dsweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DSWEEP_D:
        tag = f"synth_n{DSWEEP_N}_d{d}"
        path = out / f"{tag}.npy"
        if not path.exists():
            X = make_matrix(DSWEEP_N, d, np.random.default_rng([DSWEEP_SEED, d]))
            np.save(path, X)
        rows.append({"tag": tag, "dataset": "synthetic", "gene_set": f"d{d}", "mode": "dsweep",
                     "n_rows": DSWEEP_N, "n_cols": d})
        print(f"[dsweep] {tag}")
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)


PBMC3K_URL = ("https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
              "pbmc3k_filtered_gene_bc_matrices.tar.gz")


def make_pbmc3k() -> None:
    """10x PBMC 3k, preprocessed as in the Seurat tutorial: genes detected in >= 3 cells;
    cells with 200 < detected genes < 2500 and < 5% mitochondrial counts; log1p of counts
    per 10^4. Gives 2,638 cells x 13,714 genes, kept sparse: .npz for Python, .mtx for R."""
    import tarfile
    import urllib.request

    import scipy.io
    import scipy.sparse as sp

    out = INPUTS / "pbmc3k"
    out.mkdir(parents=True, exist_ok=True)
    tar = out / PBMC3K_URL.rsplit("/", 1)[1]
    if not tar.exists():
        print(f"[pbmc3k] downloading {PBMC3K_URL}")
        urllib.request.urlretrieve(PBMC3K_URL, tar)
    raw = out / "filtered_gene_bc_matrices" / "hg19"
    if not raw.exists():
        with tarfile.open(tar) as t:
            t.extractall(out, filter="data")

    M = scipy.io.mmread(raw / "matrix.mtx").tocsr().T.tocsr().astype(np.float64)   # cells x genes
    genes = pd.read_csv(raw / "genes.tsv", sep="\t", header=None)[1].to_numpy(str)
    enough = np.asarray((M > 0).sum(axis=1)).ravel() >= 200                # min.features = 200
    keep_g = np.asarray((M > 0).sum(axis=0)).ravel() >= 3                  # min.cells = 3
    M, genes = M[:, keep_g], genes[keep_g]
    nfeat = np.asarray((M > 0).sum(axis=1)).ravel()
    total = np.asarray(M.sum(axis=1)).ravel()
    pct_mt = 100 * np.asarray(M[:, np.char.startswith(genes, "MT-")].sum(axis=1)).ravel() / total
    keep_c = enough & (nfeat > 200) & (nfeat < 2500) & (pct_mt < 5)
    L = (sp.diags(1e4 / total[keep_c]) @ M[keep_c]).tocsr()
    L.data = np.log1p(L.data)

    tag = "pbmc3k_lognorm"
    sp.save_npz(out / f"{tag}.npz", L)
    scipy.io.mmwrite(out / f"{tag}.mtx", L, precision=17)
    pd.DataFrame([{"tag": tag, "dataset": "pbmc3k", "gene_set": "all", "mode": "pbmc3k",
                   "n_rows": L.shape[0], "n_cols": L.shape[1]}]).to_csv(out / "manifest.csv", index=False)
    print(f"[pbmc3k] {tag}: {L.shape[0]} cells x {L.shape[1]} genes, {L.nnz} non-zeros")


def load_input(path):
    """An input as a dense float array, whichever format it was saved in."""
    path = Path(path)
    if path.suffix == ".npz":
        import scipy.sparse as sp
        return sp.load_npz(path).toarray()
    return np.load(path)


def corner(path, k=3):
    """Top-left k x k block of an input as NumPy reads it (worker_r.R checks R sees the same)."""
    path = Path(path)
    if path.suffix == ".npz":
        import scipy.sparse as sp
        return sp.load_npz(path)[:k, :k].toarray().tolist()
    return np.asarray(np.load(path, mmap_mode="r")[:k, :k]).tolist()


if __name__ == "__main__":
    suites = sys.argv[1:] or ["realdata", "dsweep"]
    for s in suites:
        {"realdata": make_realdata, "dsweep": make_dsweep, "pbmc3k": make_pbmc3k}[s]()
