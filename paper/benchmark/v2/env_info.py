"""Record the benchmark environment as env.json.

Hardware, OS, power source, Python/NumPy and their BLAS, the R side (version, BLAS,
package versions, RcppML commit) and the git revision of the nnPCApy solver under test.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common import REPO, parse_result, run_child, thread_env, worker_cmd


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cpu_name() -> str:
    if os.name == "nt":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def collect(threads: int = 1) -> dict:
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info

    batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    env = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": _cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "on_ac_power": None if batt is None else bool(batt.power_plugged),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        # Library only: this parent process is not thread-pinned. The threads each cell
        # actually used are in the `blas` column of cells.csv, reported by the worker.
        "python_blas": [{k: d.get(k) for k in ("internal_api", "version", "filepath")}
                        for d in threadpool_info()],
        "thread_env": {k: v for k, v in thread_env(threads).items() if k.endswith("_THREADS")},
        "nnpcapy_git_head": _git("rev-parse", "HEAD"),
        "nnpcapy_solver_dirty": bool(_git("status", "--porcelain", "--", "src/nnpcapy")),
        "nnpcapy_solver_blob": _git("hash-object", str(REPO / "src" / "nnpcapy" / "nsprcomp.py")),
    }
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "env.json"
        spec.write_text(json.dumps({"mode": "env"}))
        out = run_child(worker_cmd("r_env", spec), thread_env(threads), 300, Path(tmp) / "env")
        env["r"] = parse_result(out["stdout"]) or {"error": out["stderr"][-2000:]}
    return env


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
