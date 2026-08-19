"""
emtscore
========

Python port of the R EMTscore package.

Three scoring modules are exposed, all reusable across notebooks:

    nsprcomp  : non-negative sparse PCA solver (`nsprcomp`, `compute_M1_M2_scores`)
    nnpca     : gene-signature scoring via non-negative sparse PCA
                (`run_nnPCA`, `execute_nnPCA_single`, `parse_gmt`, `get_nnPCA_result`)
    aucell    : AUCell-style enrichment (`execute_aucell`, `execute_aucell_single`)
    ssGSEA    : single-sample GSEA (`execute_ssgsva`, `execute_ssgsea_single`)

"""

# Light core: scoring only (numpy / scipy / pandas). Heavy submodules that need
# the optional `viz` (matplotlib, seaborn) or `sc` (anndata, scanpy, scikit-learn)
# extras are imported lazily via __getattr__ below, so `import emtscore` stays cheap.
from . import nsprcomp, nnpca, aucell, ssGSEA

from .nsprcomp import nsprcomp as nnpca_solver, compute_M1_M2_scores
from .nnpca    import run_nnPCA, execute_nnPCA_single
from .aucell   import execute_aucell, execute_aucell_single
from .ssGSEA   import execute_ssgsva, execute_ssgsea_single

_LAZY = {"pipeline", "workflow", "plotdata", "plots_em", "plots_heatmap",
         "plots_cook", "sc", "single_cell", "pathways", "scoring", "inputs"}
# convenience functions surfaced at package level but living in a heavy submodule
_LAZY_FUNCS = {"score_signature": "single_cell", "top_loading_genes": "single_cell",
               "log_cpm": "single_cell", "plot_em_scatter": "single_cell",
               "plot_signature_heatmap": "single_cell"}


def __getattr__(name):  # PEP 562: lazy access to heavy submodules / functions
    import importlib
    if name in _LAZY:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY_FUNCS:
        return getattr(importlib.import_module(f".{_LAZY_FUNCS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "nsprcomp", "nnpca", "aucell", "ssGSEA", "pipeline", "workflow", "single_cell",
    "nnpca_solver", "compute_M1_M2_scores",
    "run_nnPCA", "execute_nnPCA_single",
    "execute_aucell", "execute_aucell_single",
    "execute_ssgsva", "execute_ssgsea_single",
    "score_signature", "top_loading_genes", "log_cpm",
    "plot_em_scatter", "plot_signature_heatmap",
]

__version__ = "0.1.0"
