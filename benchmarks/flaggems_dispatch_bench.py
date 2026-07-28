# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlagGems dispatch-overhead microbenchmark -- driver.

Runs the worker once per backend in its own subprocess (routing is fixed at
import), then prints a side-by-side table. The key column is the host-side
submit-time delta (flaggems_python - cuda): the ceiling on what a C++ (kFlagOs)
dispatch could recover, since both launch the same class of GPU kernel.

    cuda      : default vendor conf (backends_cuda.conf), pure C++ dispatch
    flaggems  : FLAGOS_USE_FLAGGEMS=1 (backends_flaggems.conf), Python dispatch

Usage:
    python benchmarks/flaggems_dispatch_bench.py
    python benchmarks/flaggems_dispatch_bench.py --submit 500 --e2e 200
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKER = Path(__file__).resolve().parent / "flaggems_dispatch_overhead.py"


def _run(backend, env_extra, submit, e2e, warmup):
    env = os.environ.copy()
    env.update(env_extra)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "--backend",
            backend,
            "--submit",
            str(submit),
            "--e2e",
            str(e2e),
            "--warmup",
            str(warmup),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"worker for backend={backend} failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _index(blob):
    return {r["op"]: r for r in blob["results"]}


def _fmt(v):
    return f"{v:8.1f}" if isinstance(v, (int, float)) else f"{'ERR':>8}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", type=int, default=300)
    ap.add_argument("--e2e", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cuda = _index(_run("cuda", {}, args.submit, args.e2e, args.warmup))
    gems = _index(
        _run(
            "flaggems", {"FLAGOS_USE_FLAGGEMS": "1"}, args.submit, args.e2e, args.warmup
        )
    )

    print()
    print("Per-op host submit time and end-to-end latency (microseconds).")
    print("submit = dispatch cost (kernel time hidden by async queue);")
    print("host_delta = flaggems_submit - cuda_submit = C++ dispatch headroom.\n")
    hdr = (
        f"{'op':22} {'cat':11} {'cuda_sub':>9} {'gems_sub':>9} "
        f"{'host_delta':>10} {'cuda_e2e':>9} {'gems_e2e':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for op in cuda:
        c, g = cuda[op], gems.get(op, {})
        cs, gs = c.get("submit_us"), g.get("submit_us")
        delta = (
            (gs - cs)
            if isinstance(cs, (int, float)) and isinstance(gs, (int, float))
            else "ERR"
        )
        print(
            f"{op:22} {c.get('category', ''):11} {_fmt(cs)} {_fmt(gs)} "
            f"{_fmt(delta)} {_fmt(c.get('e2e_us'))} {_fmt(g.get('e2e_us'))}"
        )


if __name__ == "__main__":
    main()
