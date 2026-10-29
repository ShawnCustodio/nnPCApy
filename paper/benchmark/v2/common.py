"""Shared paths, timing policy and the per-process peak-memory probe for the v2 benchmark.

Every benchmark cell runs in its own fresh process (see run.py). Peak memory is read
by the *parent* from the operating system after the child exits, so R and Python are
measured the same way:

* Windows: PeakWorkingSetSize / PeakPagefileUsage of the child (GetProcessMemoryInfo).
* POSIX:   ru_maxrss of the child from os.wait4().
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = (Path(os.environ["NNPCAPY_REPO"]) if "NNPCAPY_REPO" in os.environ
        else next(p for p in HERE.parents if (p / "pyproject.toml").exists()))
BENCH = REPO / "paper" / "benchmark"     # v1 benchmark: datasets.py, synth/
INPUTS = HERE / "_inputs"                # v2-only inputs (synthetic sweep); gitignored
RESULTS = HERE / "results"

# R is called as Rterm.exe, not Rscript.exe: on Windows Rscript.exe starts Rterm.exe as a
# child process, so its own peak memory would only be the launcher's.
RTERM = os.environ.get("NNPCA_RTERM", r"D:\R\bin\x64\Rterm.exe" if os.name == "nt" else "R")

# Timing policy. worker_py.py mirrors bench::mark(min_iterations, max_iterations, min_time)
# in worker_r.R: at least MIN_REPS timed runs, then keep going until MIN_TIME_S of total
# timed run time or MAX_REPS runs, whichever comes first. One untimed warm-up precedes.
MIN_REPS, MAX_REPS, MIN_TIME_S = 5, 50, 0.5
SEED = 20261029

R_SOLVERS = ("nsprcomp", "nsprcomp_crossprod", "rcppml")
PY_SOLVERS = ("V2", "nnpcapy")


def thread_env(threads: int) -> dict:
    """Environment for a child process with every BLAS/OpenMP pool pinned to `threads`."""
    env = dict(os.environ)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    return env


def worker_cmd(solver: str, spec_path: Path) -> list[str]:
    if solver in R_SOLVERS or solver == "r_env":
        return [RTERM, "--no-echo", "--no-restore", "--no-save",
                f"--file={HERE / 'worker_r.R'}", "--args", str(spec_path)]
    return [sys.executable, str(HERE / "worker_py.py"), str(spec_path)]


if os.name == "nt":
    import ctypes
    import ctypes.wintypes as wt

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    def _peak_mb(proc: subprocess.Popen) -> tuple[float, float]:
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(int(proc._handle)), ctypes.byref(c), c.cb):
            return float("nan"), float("nan")
        return c.PeakWorkingSetSize / 2**20, c.PeakPagefileUsage / 2**20


def run_child(cmd: list[str], env: dict, timeout: float, log_prefix: Path) -> dict:
    """Run one worker; return its stdout/stderr, exit code, wall time and peak memory (MB).

    peak_ws_mb is the peak resident set (Windows peak working set / POSIX ru_maxrss);
    peak_private_mb is peak committed private memory (Windows only).
    """
    out_path, err_path = log_prefix.with_suffix(".out"), log_prefix.with_suffix(".err")
    t0 = time.perf_counter()
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env)
        if os.name == "nt":
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
            peak_ws, peak_priv = _peak_mb(proc)
        else:
            deadline = time.monotonic() + timeout
            while True:
                pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic() > deadline:
                    proc.kill()
                    pid, status, ru = os.wait4(proc.pid, 0)
                    break
                time.sleep(0.01)
            proc.returncode = rc = os.waitstatus_to_exitcode(status)
            # ru_maxrss is in bytes on macOS and KiB on Linux
            peak_ws = ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2**20
            peak_priv = float("nan")
    return {"returncode": rc, "wall_s": time.perf_counter() - t0,
            "peak_ws_mb": peak_ws, "peak_private_mb": peak_priv,
            "stdout": out_path.read_text(errors="replace"),
            "stderr": err_path.read_text(errors="replace")}


def parse_result(stdout: str) -> dict | None:
    import json
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None
