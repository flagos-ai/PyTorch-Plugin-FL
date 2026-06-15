"""Micro-benchmark: argmax latency on vocab-sized logits.

Usage:
    # torch_fl
    CUDA_VISIBLE_DEVICES=2 FLAGOS_BACKEND_CONFIG=torch_fl/backends_cuda.conf \
        python tests/perf/bench_argmax.py

    # CUDA baseline
    CUDA_VISIBLE_DEVICES=2 python tests/perf/bench_argmax.py --baseline
"""

import argparse
import time

import torch


def bench_argmax(device: torch.device, warmup: int = 50, rounds: int = 200):
    # Simulate logits shape: (1, vocab_size) typical for greedy decoding
    vocab_size = 152064  # Qwen3 vocab
    x = torch.randn(1, vocab_size, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        torch.argmax(x, dim=-1)
    torch.cuda.synchronize(device)

    # Benchmark
    times = []
    for _ in range(rounds):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch.argmax(x, dim=-1)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    print(f"argmax({1}x{vocab_size}) on {device}")
    print(f"  Median: {median:.3f} ms")
    print(f"  P10:    {p10:.3f} ms")
    print(f"  P90:    {p90:.3f} ms")
    print(f"  Min:    {times[0]:.3f} ms")
    print(f"  Max:    {times[-1]:.3f} ms")
    return median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Use native CUDA")
    args = parser.parse_args()

    if args.baseline:
        device = torch.device("cuda:0")
        print("=== CUDA Baseline ===")
    else:
        import torch_fl  # noqa: F401

        device = torch.device("flagos:0")
        print("=== torch_fl (flagos) ===")

    bench_argmax(device)


if __name__ == "__main__":
    main()
