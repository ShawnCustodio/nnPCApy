# Benchmark — R vs Python nnPCA

Side-by-side timing harness for `nsprcomp` (R, original) vs `nnpcapy.nsprcomp`
(this package). Methodology, full instructions, and headline numbers are in
the [top-level README](../README.md#benchmark--python-vs-r).

## Files

| File | What it is |
|---|---|
| `datasets.py` | Builds 47 cached input matrices from real EMT signature + expression data |
| `bench_python.py` | Times nnpcapy.nsprcomp 705 times (47 × 3 ncomps × 5 trials) |
| `bench_r.R` | R-side mirror; same inputs, same algorithm, R `nsprcomp` |
| `compare.py` | Merges both timing CSVs, writes `results/summary.csv`, emits plots |
| `results/*.csv` | Committed timing results from the most recent run |
| `plots/*.png` | Committed comparison plots |

## Quick run

```bash
python datasets.py            # ~5s — builds _inputs/*.npy
python bench_python.py        # ~3 min — writes results/python_timings.csv
# In RStudio: open bench_r.R, click Source — ~8 min, writes results/r_timings.csv
python compare.py             # ~2s — writes summary.csv + 3 PNGs in plots/
```

The cached `_inputs/` folder is gitignored (~130 MB) because it's
regenerable from the source CSVs.
