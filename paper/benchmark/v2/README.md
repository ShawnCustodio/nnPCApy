# Benchmark v2

A re-measurement of the R-vs-Python comparison in `../` with a stricter methodology. The
v1 scripts (`../bench_python.py`, `../bench_r.R`) and their committed numbers
(`../results/summary.csv`) are left untouched for comparison.

What changed from v1:

| v1 | v2 |
|---|---|
| R read the single-cell inputs scrambled (column-major `.npy`, which RcppCNPy misreads) | Inputs saved in C order; `worker_r.R` stops if R sees a different matrix than NumPy saved |
| R timed with `proc.time()` (10 ms resolution; 18/141 R medians were 0 s) | `bench::mark` in R, `time.perf_counter` in Python, same repetition policy |
| Memory: R `gc()` heap vs Python `tracemalloc` | Peak working set of each cell's own fresh process, read by the parent from the OS, minus a load-only baseline |
| Threads and BLAS not recorded | Every BLAS/OpenMP pool pinned per cell; `env.json` and per-cell `blas` record what was used |
| No iteration counts | EM iterations recorded for every solver (untimed, instrumented warm-up) |
| R nsprcomp vs nnPCApy only | Also `nsprcomp` with a `crossprod` change in its EM step, V2 (NumPy port of R's algorithm without the covariance reformulation) and RcppML |
| Leading loading only | Every component checked against nsprcomp: negative loadings, \|cos\|, principal angles, adjusted variance |
| D ≤ 446 | Synthetic sweep to D = 5,000 |

## Solvers

| name | language | what it is |
|---|---|---|
| `nsprcomp` | R | CRAN nsprcomp 0.5.1-2, unmodified (the reference) |
| `nsprcomp_crossprod` | R | the same, with `t(Xp) %*% (omega * z)` → `crossprod(Xp, omega * z)` inside `empca()`, patched in the worker process only |
| `rcppml` | R | `RcppML::pca(t(X), k, nonneg = c(TRUE, FALSE))`, GitHub commit `49df8da`, package defaults |
| `rcppml_5start` | R | the same from 5 random starts, keeping the fit with the largest Σd² (the effort of nsprcomp's default `nrestart = 5`; RcppML has no restart option, so whole fits are restarted) |
| `V2` | Python | `../synth/variants.V2_twopass`: R's data-matrix EM in NumPy, R defaults, two-pass renormalization |
| `nnpcapy` | Python | `src/nnpcapy` as checked out; `env.json` records the git revision and whether the solver file was modified |

## Requirements

- R ≥ 4.5 with `nsprcomp`, `bench`, `RcppCNPy`, `jsonlite`, and RcppML from GitHub:
  `remotes::install_github("zdebruine/RcppML@49df8da602fac19aa3932623ef1932723a978a01")`
  (needs Rtools on Windows).
- Python with `numpy`, `pandas`, `psutil`, `threadpoolctl`.
- R is started as `Rterm` (Windows: `D:\R\bin\x64\Rterm.exe`); override with `NNPCA_RTERM`.
  Rterm, not Rscript, because on Windows Rscript runs R in a child process whose memory
  the parent would not see.

## Run

```
python make_inputs.py                       # realdata (reuses ../_inputs) + dsweep
python make_inputs.py pbmc3k                # downloads 10x PBMC 3k (~7 MB) on first use
python run.py --suite realdata              # 47 inputs x ncomp 1-3 x 5 solvers, 1 thread
python run.py --suite dsweep --ncomps 1,3
python run.py --suite pbmc3k --ncomps 16 --solvers nnpcapy,rcppml --threads 1,2 --run-id pbmc3k
python run.py --suite pbmc3k --ncomps 16 --solvers nsprcomp --max-reps 0 --run-id pbmc3k
python analyze.py <run-id>                  # run-id is printed by run.py
python second_seed.py realdata_89c9f66_r1 --old-run realdata_head3b31cf9_t1_corder
                                            # R refit with a second seed; Table S1B; about 2 min
python plot_fig2.py                         # Fig 2 (fig2, figR1) from the run folders; seconds
```

`run.py` resumes an interrupted run when given the same `--run-id`, which also lets one run
combine solvers timed with different settings, as above. `--threads 1,2` adds a
2-thread configuration. `--max-reps 0` times a single fit (the instrumented warm-up)
instead of at least five, for cells that take many minutes. `--densify` gives Python
solvers dense input, e.g. to time a solver revision from before sparse support (set
`NNPCAPY_REPO` to that checkout).

The pbmc3k suite is the whole transcriptome of 10x PBMC 3k after the Seurat tutorial's
filtering (2,638 cells x 13,714 genes, 6% non-zero). Python solvers get the sparse
matrix (`nnpcapy`) or a dense copy (`V2`); in R, RcppML gets it as a sparse `dgCMatrix`
and nsprcomp as a dense matrix, whose copy counts towards the fit's memory.

On a laptop: plug in to AC power (recorded in `env.json`), use the best-performance power
mode, and close other heavy programs. Solvers are run in a shuffled order per input so
that slow drift affects all of them alike.

## Run times

Measured on the laptop used for the paper (Intel i7-1165G7, AC power, 1 thread per fit), as
the sum of the worker processes' wall times:

| run | figure | time |
|---|---|---|
| `--suite realdata` (47 inputs x ncomp 1-3, all solvers) | Fig 2A-B | about 35 min per repeat; Fig 2 uses 3 repeats |
| `--suite dsweep --ncomps 1,3` | Fig 2C | about 35 min |
| `--suite pbmc3k` (both commands above) | Fig 2D | about 30 min; one R nsprcomp fit alone takes 17 min and about 2 GB |
| `plot_fig2.py` | Fig 2 | seconds |
| `second_seed.py` (runs `second_seed_r.R`) | Supplementary Table S1B | about 2 min (141 R fits) |

R nsprcomp takes most of each suite's time. Redrawing the figures needs only the run folders
in `results/`, not a rerun.

## Method details

- **One process per cell.** Load input, one untimed warm-up fit (seeded with `SEED` in
  `common.py`, instrumented for iteration counts, its loadings saved), then timed fits: at
  least 5, continuing until 0.5 s of timed runs or 50 runs. Median, min and max are reported.
- **Memory.** `peak_ws_mb` is the peak working set (Windows) or `ru_maxrss` (POSIX) of the
  whole worker process. A `baseline` cell per input and solver loads the same packages and
  input without fitting; `analyze.py` reports the difference as `mem_delta_mb`. Both
  languages read the same `.npy` file and keep the fitted result as an object.
- **Iterations.** R: `trace()` on nsprcomp's internal `empca()` reads its loop counter at
  exit. Python: a profile hook reads the EM loop counter when the EM function returns.
  RcppML: `misc$iters_per_factor`. One count per EM call: nsprcomp-style solvers make
  restarts × 2 passes × components calls, RcppML one per component (× 5 for `rcppml_5start`).
- **Accuracy** (`analyze.py`) is computed from the saved warm-up loadings against
  nsprcomp's on the same input, k and threads, with one adjusted-variance convention for
  every solver.

## Not yet included

- Absolute wall time of one complete bootstrap analysis.
- Figures.

## Outputs

`results/<run-id>/`: `env.json`, `cells.csv`, `summary_cells.csv`, `summary_overall.csv`,
`loadings/`, `logs/`, `specs/`. All of `results/` and `_inputs/` is gitignored until we
decide which summaries to publish.
