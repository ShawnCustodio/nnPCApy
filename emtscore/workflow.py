"""
emtscore.workflow - thin facade re-exporting from focused sub-modules
"""

from .inputs import Inputs, load_inputs
from .pathways import plot_top_pathways, run_pathway_correlation_v2
from .plotdata import PlotData, RebuildResult, prepare_plot_dataframes, rebuild_em_for_plot
from .plots_cook import (
    compute_stem_senescence,
    plot_em_pc_panels_cook,
    plot_em_vs_stem_sen,
    plot_stemness_vs_senescence,
)
from .plots_em import (
    plot_combined_em_m1_m2,
    plot_em_panel,
    plot_em_section,
    plot_m1_histogram,
    plot_m1_m2,
)
from .plots_heatmap import plot_full_m_heatmap, plot_pc_driver_heatmap
from .sc import build_gmm_in_em_space, load_cook_adatas, plot_emt_vs_pseudotime, plot_gmm_sankey
from .scoring import Scores, score_all_methods

__all__ = [
    "Inputs",
    "PlotData",
    "RebuildResult",
    "Scores",
    "build_gmm_in_em_space",
    "compute_stem_senescence",
    "load_cook_adatas",
    "load_inputs",
    "plot_combined_em_m1_m2",
    "plot_em_panel",
    "plot_em_pc_panels_cook",
    "plot_em_section",
    "plot_em_vs_stem_sen",
    "plot_emt_vs_pseudotime",
    "plot_full_m_heatmap",
    "plot_gmm_sankey",
    "plot_m1_histogram",
    "plot_m1_m2",
    "plot_pc_driver_heatmap",
    "plot_stemness_vs_senescence",
    "plot_top_pathways",
    "prepare_plot_dataframes",
    "rebuild_em_for_plot",
    "run_pathway_correlation_v2",
    "score_all_methods",
]
