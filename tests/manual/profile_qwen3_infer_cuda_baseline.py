"""
Profile Qwen3 inference on native CUDA to provide op-level baseline.

Same profiling methodology as profile_qwen3_infer.py but runs on CUDA.
Compare outputs to identify which ops are slow in torch_fl.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tests/manual/profile_qwen3_infer_cuda_baseline.py
"""

import argparse
import time
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpProfiler(TorchDispatchMode):
    """Profile op calls and execution time."""

    def __init__(self, device):
        self.device = device
        self.op_counts = defaultdict(int)
        self.op_times = defaultdict(float)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)
        self.op_counts[op_name] += 1

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0
        self.op_times[op_name] += elapsed

        return result

    def report(self):
        print("\n=== Op Profile Summary (CUDA baseline) ===")
        print(f"Total unique ops: {len(self.op_counts)}")
        print(f"Total op calls: {sum(self.op_counts.values())}")
        print(f"Total op time: {sum(self.op_times.values()):.3f}s")

        print("\n=== Top 20 slowest ops (by total time) ===")
        for op, total_time in sorted(
            self.op_times.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            count = self.op_counts[op]
            avg_time_us = (total_time / count) * 1e6
            pct = (total_time / sum(self.op_times.values())) * 100
            print(
                f"{op:50s} {total_time:6.3f}s ({pct:5.1f}%)  "
                f"{count:5d} calls  {avg_time_us:7.1f}µs/call"
            )

        print("\n=== Top 20 most frequent ops ===")
        for op, count in sorted(
            self.op_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            total_time = self.op_times[op]
            avg_time_us = (total_time / count) * 1e6
            print(f"{op:50s} {count:5d} calls  {avg_time_us:7.1f}µs/call")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B", help="Path to model"
    )
    parser.add_argument("--tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()
    device = args.device

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    model.model.layers[0].self_attn.config._attn_implementation = "eager"
    print("Model loaded, attention: eager")

    # Prepare input
    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language model.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print()

    # Warmup
    print("Warmup...")
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=args.tokens)
    torch.cuda.synchronize(device)
    print(f"Warmup: {time.time() - t0:.2f}s\n")

    # Profiled run
    print("Profiling...")
    profiler = OpProfiler(device)
    torch.cuda.synchronize(device)
    t0 = time.time()
    with profiler:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.tokens)
    torch.cuda.synchronize(device)
    profiled_time = time.time() - t0

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tps = new_tokens / profiled_time
    print(f"\nProfiled run: {profiled_time:.2f}s, {new_tokens} tokens, {tps:.2f} tok/s")

    profiler.report()

    # Per-token breakdown
    total_op_time = sum(profiler.op_times.values())
    print("\n=== Per-Token Breakdown ===")
    print(f"Wall-clock time:     {profiled_time:.3f}s")
    print(f"Op execution time:   {total_op_time:.3f}s")
    print(f"Unaccounted overhead: {profiled_time - total_op_time:.3f}s")
    print(f"Time per token: {profiled_time / new_tokens * 1000:.1f}ms")
    print(f"Ops per token:  {sum(profiler.op_counts.values()) / new_tokens:.0f}")
    avg_op_time_us = (total_op_time / sum(profiler.op_counts.values())) * 1e6
    print(f"Avg op time:    {avg_op_time_us:.1f}µs")


if __name__ == "__main__":
    main()
