"""
Count actual tensor allocations per token during Qwen3 inference.
"""
import torch
import time
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).cuda().eval()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = 64

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

    # Count allocations via torch.cuda.memory_stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    stats_before = torch.cuda.memory_stats()
    alloc_before = stats_before["allocation.all.current"]

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

    torch.cuda.synchronize()
    stats_after = torch.cuda.memory_stats()

    # num_alloc_retries won't help, use segment info
    # Better: use memory_stats "allocation.all.allocated" which counts total allocs
    total_allocs = stats_after["allocation.all.allocated"] - stats_before["allocation.all.allocated"]
    total_frees = stats_after["allocation.all.freed"] - stats_before["allocation.all.freed"]

    print(f"=== Allocation count during {num_tokens}-token generation ===")
    print(f"Total allocations: {total_allocs}")
    print(f"Total frees:       {total_frees}")
    print(f"Allocs per token:  {total_allocs / num_tokens:.1f}")
    print(f"Frees per token:   {total_frees / num_tokens:.1f}")
    print()
    print(f"Per-token allocator overhead estimate:")
    print(f"  @ 506ns/alloc saving: {total_allocs / num_tokens * 506 / 1e6:.3f} ms/token")


if __name__ == "__main__":
    main()
