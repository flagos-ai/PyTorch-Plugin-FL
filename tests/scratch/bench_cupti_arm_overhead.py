"""Finding 6 probe: cost of arming CUPTI RUNTIME + EXTERNAL_CORRELATION at import.

Times fixed non-profiling workloads so the overhead imposed on users who merely
`import torch_fl` can be measured. Run under both builds and compare.

Two workloads, because they stress different things:
  * "gemm"   -- 1024x1024 matmuls: GPU-bound, so per-launch CPU cost is hidden.
  * "launch" -- tiny 32x32 ops: launch-bound, ~all CPU time in the CUDA runtime
                API, which is exactly what CUPTI_ACTIVITY_KIND_RUNTIME
                instruments. This is the workload that can actually expose the
                regression the reviewer is worried about.
"""

import statistics
import sys
import time

import torch
import torch_fl  # noqa: F401


def bench_gemm(reps=12, iters=50, size=1024):
    x = torch.randn(size, size, device="flagos:0")
    y = torch.randn(size, size, device="flagos:0")
    for _ in range(10):
        z = (x @ y).relu()
    z.sum().item()

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            z = (x @ y).relu()
        z.sum().item()  # sync
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def bench_launch(reps=12, iters=2000, size=32):
    """Launch-bound: tiny kernels, so wall time ~= CUDA runtime API cost."""
    x = torch.randn(size, size, device="flagos:0")
    for _ in range(200):
        y = x.relu()
    y.sum().item()

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            y = x.relu()
        y.sum().item()  # sync
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def report(label, samples, iters):
    mean = statistics.mean(samples)
    print(f"{label}: n={len(samples)} mean={mean:.3f}ms "
          f"stdev={statistics.stdev(samples):.3f}ms "
          f"min={min(samples):.3f}ms median={statistics.median(samples):.3f}ms "
          f"max={max(samples):.3f}ms  per_iter={mean * 1000 / iters:.3f}us")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    report(f"{label} [gemm]", bench_gemm(), 50)
    report(f"{label} [launch]", bench_launch(), 2000)
