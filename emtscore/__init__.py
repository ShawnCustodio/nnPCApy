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
from . import aucell, nnpca, nsprcomp, ssGSEA
from .aucell import execute_aucell, execute_aucell_single
from .nnpca import execute_nnPCA_single, run_nnPCA
from .nsprcomp import compute_M1_M2_scores
from .nsprcomp import nsprcomp as nnpca_solver
from .ssGSEA import execute_ssgsea_single, execute_ssgsva

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
    "aucell",
    "compute_M1_M2_scores",
    "execute_aucell",
    "execute_aucell_single",
    "execute_nnPCA_single",
    "execute_ssgsea_single",
    "execute_ssgsva",
    "log_cpm",
    "nnpca",
    "nnpca_solver",
    "nsprcomp",
    "pipeline",
    "plot_em_scatter",
    "plot_signature_heatmap",
    "run_nnPCA",
    "score_signature",
    "single_cell",
    "ssGSEA",
    "top_loading_genes",
    "workflow",
]

__version__ = "0.1.0"
