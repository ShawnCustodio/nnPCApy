"""One Python benchmark cell, in a fresh process launched by run.py.

Usage: python worker_py.py <spec.json>

spec keys: input (.npy), solver ("nnpcapy" | "V2"), ncomp, mode ("fit" | "baseline"),
seed, out_npy, min_reps, max_reps, min_time.

  nnpcapy  the solver in src/nnpcapy (whatever revision is checked out; run.py records it)
  V2       synth/variants.V2_twopass: a NumPy port of R's data-matrix EM with R's defaults
           and two-pass renormalization, without the covariance reformulation

mode "baseline" loads the same modules and input but does not fit; run.py subtracts its
peak memory from the fit cells. Prints one `RESULT {json}` line on stdout.
"""
import json
import sys
import time

spec = json.load(open(sys.argv[1]))

import numpy as np  # noqa: E402  (thread env vars are set by run.py before this import)

from common import BENCH, REPO  # noqa: E402

sys.path[:0] = [str(REPO / "src"), str(BENCH / "synth")]

X = np.load(spec["input"])
k = int(spec["ncomp"])

if spec["solver"] == "nnpcapy":
    import nnpcapy
    _solver = sys.modules["nnpcapy.nsprcomp"]
    # the EM function is _empca since the cov/data split, _empca_cov before it
    em_code = {(getattr(_solver, "_empca", None) or _solver._empca_cov).__code__}

    def fit():
        return nnpcapy.nsprcomp(X, ncomp=k, nneg=True, center=True, scale_=False)
elif spec["solver"] == "V2":
    import variants
    em_code = {variants._empca_data.__code__}

    def fit():
        return variants.V2_twopass(X, k)
else:
    raise SystemExit(f"unknown Python solver {spec['solver']!r}")

result = {"solver": spec["solver"], "mode": spec["mode"], "n_rows": X.shape[0], "n_cols": X.shape[1]}
try:
    from threadpoolctl import threadpool_info
    result["blas"] = [{k_: d.get(k_) for k_ in ("internal_api", "num_threads", "version")}
                      for d in threadpool_info()]
except ImportError:
    pass

if spec["mode"] == "fit":
    # Untimed warm-up: seeded, instrumented, and the one whose loadings are saved.
    # EM iterations are read from the loop variable `_` when the EM function returns
    # (for an EM that hits em_maxiter this reads em_maxiter - 1).
    iters = []

    def _prof(frame, event, arg):
        if event == "return" and frame.f_code in em_code:
            iters.append(int(frame.f_locals.get("_", -1)))

    np.random.seed(int(spec["seed"]))
    sys.setprofile(_prof)
    res = fit()
    sys.setprofile(None)
    np.save(spec["out_npy"], np.asarray(res["rotation"], dtype=np.float64))

    times = []
    while len(times) < spec["max_reps"] and (len(times) < spec["min_reps"] or sum(times) < spec["min_time"]):
        t0 = time.perf_counter()
        fit()
        times.append(time.perf_counter() - t0)

    result.update(times=times, iters=iters, sdev=[float(s) for s in res["sdev"]])

print("RESULT " + json.dumps(result), flush=True)
