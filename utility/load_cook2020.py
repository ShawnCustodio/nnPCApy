"""
utility/load_cook2020.py
------------------------
Loads Cook et al. 2020 single-cell RNA-seq data from CSV exports.

The loader auto-resolves the ``data/cook2020/`` directory, searching (in
order): explicit argument, bundled ``Python_Conv/data/cook2020/``, every
ancestor of the CWD, and every ancestor of the package root.  That means a
notebook does not need a ``data_dir`` kwarg; the Cook folder shipped next to
the parent repo is picked up automatically.

Datasets available:
    A549_TNF, A549_EGF, A549_TGFB1
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad


# ── Dataset registry ──────────────────────────────────────────────────────────
COOK_DATASETS = ["A549_TNF", "A549_EGF", "A549_TGFB1"]
_DEFAULT_DATA_DIR = Path("data/cook2020")


def resolve_cook_dir(data_dir: str | Path | None = None) -> Path:
    """Find the Cook 2020 data directory.

    Search order:
      1. Explicit ``data_dir`` argument (if given and contains files)
      2. ``<package>/data/cook2020/`` (bundled with Python_Conv)
      3. Walk up from CWD looking for ``data/cook2020/`` containing Cook CSVs
      4. ``PythonLab/data/cook2020/`` (repo-level parent of Python_Conv)
    """
    def _has_cook_data(p: Path) -> bool:
        if not p.exists():
            return False
        for ds in COOK_DATASETS:
            if (p / f"{ds}_counts.csv").exists():
                return True
        return False

    if data_dir is not None:
        p = Path(data_dir)
        if _has_cook_data(p):
            return p

    pkg_root = Path(__file__).resolve().parent.parent  # Python_Conv/
    bundled = pkg_root / "data" / "cook2020"
    if _has_cook_data(bundled):
        return bundled

    cwd = Path.cwd().resolve()
    for ancestor in [cwd, *cwd.parents]:
        cand = ancestor / "data" / "cook2020"
        if _has_cook_data(cand):
            return cand
        cand2 = ancestor / "cook2020"
        if _has_cook_data(cand2):
            return cand2

    for ancestor in pkg_root.parents:
        cand = ancestor / "data" / "cook2020"
        if _has_cook_data(cand):
            return cand

    return bundled


# ── Public API ────────────────────────────────────────────────────────────────

def list_cook_datasets() -> pd.DataFrame:
    """Print and return the 3 Cook et al. 2020 datasets used in Section 3."""
    df = pd.DataFrame({
        "Dataset":   COOK_DATASETS,
        "CellLine":  ["A549"] * 3,
        "Treatment": ["TNF", "EGF", "TGFB1"],
    })
    print("Cook et al. 2020 - Section 3 datasets")
    print("=" * 40)
    print(df.to_string(index=False))
    return df


def load_cook2020(dataset: str, data_dir: str | Path | None = None) -> ad.AnnData:
    """Load a single Cook et al. 2020 dataset from CSV exports.

    Parameters
    ----------
    dataset : str
        One of 'A549_TNF', 'A549_EGF', 'A549_TGFB1'.
    data_dir : str or Path, optional
        Directory containing the CSV files. If None, auto-resolved.
    """
    root = resolve_cook_dir(data_dir)
    _check_files(root, dataset)

    print(f"  Loading {dataset} from {root} ...")

    genes = pd.read_csv(root / f"{dataset}_genes.csv")["gene"].tolist()
    cells = pd.read_csv(root / f"{dataset}_cells.csv")["cell"].tolist()

    triplets = pd.read_csv(root / f"{dataset}_counts.csv")
    counts = sp.csr_matrix(
        (triplets["x"].values,
         (triplets["i"].values - 1,
          triplets["j"].values - 1)),
        shape=(len(genes), len(cells))
    ).T  # cells x genes

    metadata = pd.read_csv(root / f"{dataset}_metadata.csv", index_col=0)
    metadata.index = cells

    var_df = pd.DataFrame(index=genes)

    adata = ad.AnnData(X=counts, obs=metadata, var=var_df)
    adata.layers["counts"] = counts
    adata.obs_names = cells
    adata.var_names = genes

    adata.uns["dataset"] = dataset
    adata.uns["source"]  = "Cook et al. 2020 - Zenodo 18489669"

    print(f"  Done. {dataset}: {adata.shape}  (cells x genes)")
    return adata


def load_all_cook2020(data_dir: str | Path | None = None,
                       verbose: bool = True) -> dict[str, ad.AnnData]:
    """Load all 3 Cook et al. 2020 datasets."""
    out: dict[str, ad.AnnData] = {}
    for i, dataset in enumerate(COOK_DATASETS, 1):
        if verbose:
            print(f"\n[{i}/3] {dataset}")
        out[dataset] = load_cook2020(dataset, data_dir=data_dir)
    return out


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_files(root: Path, dataset: str) -> None:
    """Raise a clear error if any expected CSV file is missing."""
    missing = []
    for suffix in ["_counts.csv", "_genes.csv", "_cells.csv", "_metadata.csv"]:
        f = root / f"{dataset}{suffix}"
        if not f.exists():
            missing.append(str(f))
    if missing:
        raise FileNotFoundError(
            f"\nMissing files for '{dataset}':\n"
            + "\n".join(f"  {f}" for f in missing)
        )
