# Fetch the FULL Cook 2020 A549 TGF-β single-cell matrix via Bioconductor
# ExperimentHub (the route used by wenmm/EMTscore) and export it in a compact
# form this project can load for a genome-wide C2 screen.
#
# Run locally (needs internet + Bioconductor):
#   Rscript fetch_cook_A549_TGFB1.R
# Outputs (written next to this script, under ./cook_full/):
#   A549_TGFB1_counts.mtx     sparse genes x cells count matrix (Matrix Market)
#   A549_TGFB1_genes.txt      gene symbols (rows of the matrix)
#   A549_TGFB1_cells.txt      cell barcodes (columns of the matrix)
#   A549_TGFB1_meta.csv       per-cell metadata (Pseudotime, Time, Treatment, ...)

options(repos = c(CRAN = "https://cloud.r-project.org"))

options(timeout = 3600)

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
for (p in c("ExperimentHub", "EMTscoreData", "SummarizedExperiment", "Matrix")) {
  if (!requireNamespace(p, quietly = TRUE)) BiocManager::install(p, update = FALSE, ask = FALSE)
}

suppressPackageStartupMessages({
  library(ExperimentHub)
  library(Matrix)
})

eh  <- ExperimentHub::ExperimentHub()
obj <- eh[["EH10293"]]          # A549_TGFB1 (full single-cell object)
message("Loaded EH10293; class = ", paste(class(obj), collapse = ", "))

# --- extract a genes x cells count matrix + per-cell metadata, robustly ---
get_counts <- function(o) {
  if (inherits(o, "Seurat")) {
    if (!requireNamespace("SeuratObject", quietly = TRUE)) library(Seurat)
    m <- tryCatch(SeuratObject::GetAssayData(o, layer = "counts"),
                  error = function(e) SeuratObject::GetAssayData(o, slot = "counts"))
    return(list(counts = m, meta = o@meta.data))
  }
  if (inherits(o, "SingleCellExperiment") || inherits(o, "SummarizedExperiment")) {
    library(SummarizedExperiment)
    an <- if ("counts" %in% assayNames(o)) "counts" else assayNames(o)[1]
    return(list(counts = assay(o, an), meta = as.data.frame(colData(o))))
  }
  stop("Unrecognized object class: ", paste(class(o), collapse = ", "))
}

res    <- get_counts(obj)
counts <- as(res$counts, "CsparseMatrix")
meta   <- res$meta

dir.create("cook_full", showWarnings = FALSE)
Matrix::writeMM(counts, "cook_full/A549_TGFB1_counts.mtx")
writeLines(rownames(counts), "cook_full/A549_TGFB1_genes.txt")
writeLines(colnames(counts), "cook_full/A549_TGFB1_cells.txt")
write.csv(meta, "cook_full/A549_TGFB1_meta.csv", row.names = TRUE)

message(sprintf("Wrote cook_full/: %d genes x %d cells; meta cols: %s",
                nrow(counts), ncol(counts), paste(colnames(meta), collapse = ", ")))
