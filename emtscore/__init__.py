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

from . import nsprcomp, nnpca, aucell, ssGSEA, pipeline 

from .nsprcomp import nsprcomp as nnpca_solver, compute_M1_M2_scores  
from .nnpca    import run_nnPCA, execute_nnPCA_single                 
from .aucell   import execute_aucell, execute_aucell_single           
from .ssGSEA   import execute_ssgsva, execute_ssgsea_single          

__all__ = [
    "nsprcomp", "nnpca", "aucell", "ssGSEA", "pipeline",
    "nnpca_solver", "compute_M1_M2_scores",
    "run_nnPCA", "execute_nnPCA_single",
    "execute_aucell", "execute_aucell_single",
    "execute_ssgsva", "execute_ssgsea_single",
]

__version__ = "0.1.0"
