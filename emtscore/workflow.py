"""
emtscore.workflow - thin facade re-exporting from focused sub-modules
"""

from .inputs       import Inputs, load_inputs
from .scoring      import Scores, score_all_methods
from .plotdata     import PlotData, prepare_plot_dataframes, RebuildResult, rebuild_em_for_plot
from .plots_em     import plot_em_panel, plot_em_section, plot_m1_m2, plot_combined_em_m1_m2, plot_m1_histogram
from .plots_heatmap import plot_full_m_heatmap, plot_pc_driver_heatmap
from .sc           import load_cook_adatas, build_gmm_in_em_space, plot_emt_vs_pseudotime, plot_gmm_sankey
from .pathways     import run_pathway_correlation_v2, plot_top_pathways
from .plots_cook   import (
    plot_em_pc_panels_cook,
    compute_stem_senescence,
    plot_stemness_vs_senescence,
    plot_em_vs_stem_sen,
)

__all__ = [
    "Inputs", "load_inputs",
    "Scores", "score_all_methods",
    "PlotData", "prepare_plot_dataframes", "RebuildResult", "rebuild_em_for_plot",
    "plot_em_panel", "plot_em_section", "plot_m1_m2", "plot_combined_em_m1_m2", "plot_m1_histogram",
    "plot_full_m_heatmap", "plot_pc_driver_heatmap",
    "load_cook_adatas", "build_gmm_in_em_space", "plot_emt_vs_pseudotime", "plot_gmm_sankey",
    "run_pathway_correlation_v2", "plot_top_pathways",
    "plot_em_pc_panels_cook", "compute_stem_senescence",
    "plot_stemness_vs_senescence", "plot_em_vs_stem_sen",
]
