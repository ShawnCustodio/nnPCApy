"""
nnPCApy — non-negative sparse PCA (nsprcomp) ported from R, plus the
companion AUCell and ssGSEA scoring methods used by the EMTscore
reference pipeline.

Public API:
    from nnpcapy import nsprcomp
"""

from .nsprcomp import nsprcomp

__all__ = ["nsprcomp"]
__version__ = "0.1.0"
