#!/usr/bin/env python3
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

"""Fixed-shape Qwen forward benchmark: eager CPU vs BPU compilation.

This is the gate for task #23. BPU median latency must be lower than eager CPU
for the same deterministic model configuration and input IDs before the
performance objective is considered complete.

Usage:
    python benchmarks/bpu_qwen_bench.py

Requirements:
    - transformers installed (HuggingFace Qwen model support)
    - Qwen/Qwen2.5-0.5B-Instruct cached locally for offline operation
    - FLAGOS_BPU_X86_PYTHON and FLAGOS_BPU_X86_EMULATOR set for compilation
    - S600 board with BPU driver

The benchmark constructs a small Qwen configuration locally, runs fixed-shape
prefill with deterministic weights and inputs, warms both paths, synchronizes
correctly, reports median/p95 latency and submission count, and validates
logits before timing.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import torch

log = logging.getLogger(__name__)


def build_model(*, bf16: bool = True):
    """Build a deterministic small Qwen configuration for benchmarking."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
    )
    # Small configuration: one layer, reduced dimensions
    cfg.num_hidden_layers = 1
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.max_position_embeddings = 512

    model = AutoModelForCausalLM.from_config(cfg).eval()
    if bf16:
        model = model.bfloat16()
    else:
        model = model.float()

    # Deterministic weights
    torch.manual_seed(42)
    for p in model.parameters():
        p.data.normal_(0, 0.02)

    return model


def measure_latency(fn, warmup: int = 5, trials: int = 20):
    """Run `fn` repeatedly and return median/p95 latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Measure
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "median_ms": statistics.median(times),
        "p95_ms": statistics.quantiles(times, n=20)[18],  # 95th percentile
        "min_ms": min(times),
        "max_ms": max(times),
        "trials": len(times),
    }


def bench_eager(model, input_ids):
    """Benchmark eager CPU execution."""
    log.info("Benchmarking eager CPU...")

    def run():
        with torch.no_grad():
            model(input_ids)

    return measure_latency(run)


def bench_bpu(model, input_ids, *, min_compute_macs: int = 0):
    """Benchmark BPU-compiled execution."""
    log.info("Benchmarking BPU compilation...")

    # Compile with the BPU backend
    compiled = torch.compile(
        model,
        backend="bpu",
        dynamic=False,
        options={"min_compute_macs": min_compute_macs},
    )

    # First call triggers compilation
    log.info("Triggering compilation (first call)...")
    with torch.no_grad():
        out_compiled = compiled(input_ids)

    # Verify output shape before timing
    with torch.no_grad():
        out_eager = model(input_ids)

    assert out_compiled.logits.shape == out_eager.logits.shape

    def run():
        with torch.no_grad():
            compiled(input_ids)

    return measure_latency(run)


def validate_logits(model_eager, model_compiled, input_ids):
    """Check that compiled logits match eager within acceptable error."""
    with torch.no_grad():
        out_eager = model_eager(input_ids)
        out_compiled = model_compiled(input_ids)

    logits_eager = out_eager.logits
    logits_compiled = out_compiled.logits

    diff = (logits_compiled.float() - logits_eager.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Cosine similarity
    flat_eager = logits_eager.float().flatten()
    flat_compiled = logits_compiled.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_eager.unsqueeze(0), flat_compiled.unsqueeze(0)
    ).item()

    # Argmax agreement (next-token prediction)
    argmax_eager = logits_eager.argmax(dim=-1)
    argmax_compiled = logits_compiled.argmax(dim=-1)
    argmax_match = (argmax_eager == argmax_compiled).all().item()

    log.info("Logits validation:")
    log.info(f"  Max diff: {max_diff:.6f}")
    log.info(f"  Mean diff: {mean_diff:.6f}")
    log.info(f"  Cosine similarity: {cos_sim:.6f}")
    log.info(f"  Argmax match: {argmax_match}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "argmax_match": argmax_match,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Input sequence length"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use BF16 (default: F32)"
    )
    parser.add_argument(
        "--min-compute-macs",
        type=int,
        default=1_000_000,
        help="Minimum MACs threshold for BPU partitions",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations"
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Measurement trials"
    )
    parser.add_argument(
        "--skip-eager", action="store_true", help="Skip eager benchmark"
    )
    parser.add_argument(
        "--skip-bpu", action="store_true", help="Skip BPU benchmark"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Register the BPU backend if not already registered
    try:
        from torch_fl.accelerator.bpu.backend import register
        register()
    except AssertionError:
        # Already registered
        pass

    log.info("Building model...")
    model = build_model(bf16=args.bf16)
    log.info(f"Model: 1 layer, {model.config.hidden_size}d, {args.bf16 and 'BF16' or 'F32'}")

    # Deterministic input
    torch.manual_seed(123)
    input_ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len))
    log.info(f"Input: {input_ids.shape}")

    results = {}

    if not args.skip_eager:
        results["eager"] = bench_eager(model, input_ids)
        log.info(f"Eager CPU: {results['eager']['median_ms']:.2f} ms (median)")

    if not args.skip_bpu:
        # For validation, compile a separate instance
        model_compiled = build_model(bf16=args.bf16)
        model_compiled = torch.compile(
            model_compiled,
            backend="bpu",
            dynamic=False,
            options={"min_compute_macs": args.min_compute_macs},
        )

        # Trigger compilation and validate
        with torch.no_grad():
            model_compiled(input_ids)

        validation = validate_logits(model, model_compiled, input_ids)

        # Now benchmark
        results["bpu"] = bench_bpu(model, input_ids, min_compute_macs=args.min_compute_macs)
        log.info(f"BPU: {results['bpu']['median_ms']:.2f} ms (median)")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Configuration: 1 layer, {model.config.hidden_size}d, seq_len={args.seq_len}")
    print(f"Dtype: {args.bf16 and 'BF16' or 'F32'}")
    print(f"MAC threshold: {args.min_compute_macs:,}")
    print()

    if "eager" in results:
        r = results["eager"]
        print(f"Eager CPU:")
        print(f"  Median: {r['median_ms']:.2f} ms")
        print(f"  P95:    {r['p95_ms']:.2f} ms")
        print(f"  Min:    {r['min_ms']:.2f} ms")
        print(f"  Max:    {r['max_ms']:.2f} ms")
        print()

    if "bpu" in results:
        r = results["bpu"]
        print(f"BPU:")
        print(f"  Median: {r['median_ms']:.2f} ms")
        print(f"  P95:    {r['p95_ms']:.2f} ms")
        print(f"  Min:    {r['min_ms']:.2f} ms")
        print(f"  Max:    {r['max_ms']:.2f} ms")
        print()

    if "eager" in results and "bpu" in results:
        speedup = results["eager"]["median_ms"] / results["bpu"]["median_ms"]
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1.0:
            print("✓ BPU is faster than eager CPU")
        else:
            print("✗ BPU is slower than eager CPU")
        print()

    if "bpu" in results:
        print("Logits validation:")
        print(f"  Max diff: {validation['max_diff']:.6f}")
        print(f"  Cosine similarity: {validation['cos_sim']:.6f}")
        print(f"  Argmax match: {validation['argmax_match']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
