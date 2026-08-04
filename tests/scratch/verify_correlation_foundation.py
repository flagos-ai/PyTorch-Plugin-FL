"""
Verification gate for profiler parity design foundation (Task 1, one-shot).

Tests 3 claims:
1. Chrome trace contains flow arrows (ac2g category, count > 0)
2. key_averages() shows aten::mm with self_device_time_total > 0
3. Runtime events appear with a recognizable runtime category

NOTE on metric 1: kineto's chrome-trace writer does NOT emit a top-level
"flowEvents" array (the brief's scaffold assumed one). Flow arrows are ordinary
entries inside "traceEvents" with ph in {"s", "f"} and cat "ac2g". We count both
places so a zero cannot be an artifact of looking in the wrong key.
"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torch_fl  # noqa: F401  (registers the flagos PrivateUse1 backend)


def run_traced_ops():
    """5x matmul+relu, same workload as spec's A/B comparison."""
    x = torch.randn(1024, 1024, device="flagos:0")
    y = torch.randn(1024, 1024, device="flagos:0")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        with_stack=False,
    ) as prof:
        for _ in range(5):
            z = (x @ y).relu()
        z.sum().item()  # force sync

    return prof


def verify_flow_arrows(trace):
    """Metric 1: ac2g flow count."""
    top_level = [e for e in trace.get("flowEvents", []) if e.get("cat") == "ac2g"]
    inline = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("cat") == "ac2g" and e.get("ph") in ("s", "f")
    ]
    total = len(top_level) + len(inline)
    print(f"Flow arrows (ac2g): {total}  "
          f"(top-level flowEvents={len(top_level)}, inline ph=s/f={len(inline)})")
    return total


def verify_device_time_attribution(prof):
    """Metric 2: aten::mm self_device_time_total."""
    key_avg = prof.key_averages()
    mm_events = [e for e in key_avg if "mm" in e.key.lower()]
    for evt in mm_events:
        print(f"  {evt.key}: self_device_time_total={evt.self_device_time_total}us "
              f"device_time_total={evt.device_time_total}us count={evt.count}")
    for evt in mm_events:
        if "aten::mm" in evt.key:
            print(f"aten::mm self_device_time_total={evt.self_device_time_total}us")
            return evt.self_device_time_total
    print("aten::mm NOT FOUND in key_averages()")
    return None


def verify_runtime_category(trace, expected_cats):
    """Metric 3: runtime events present with an acceptable category."""
    cats = Counter(e.get("cat") for e in trace.get("traceEvents", []))
    runtime_events = [
        e for e in trace.get("traceEvents", []) if e.get("cat") in expected_cats
    ]
    print(f"All trace categories: {dict(sorted(cats.items(), key=lambda kv: str(kv[0])))}")
    if not runtime_events:
        print(f"Runtime category: NONE of {sorted(expected_cats)} present")
        return None
    actual = Counter(e["cat"] for e in runtime_events)
    print(f"Runtime category: {dict(actual)} ({len(runtime_events)} events)")
    return sorted(actual)[0]


def kernel_event_summary(trace):
    """Context: how many device kernels landed, and do they carry a correlation?"""
    kernels = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]
    with_corr = [
        k for k in kernels if (k.get("args") or {}).get("correlation") not in (None, 0)
    ]
    print(f"Kernel events: {len(kernels)} (with non-zero correlation arg: {len(with_corr)})")
    if kernels:
        k = kernels[0]
        print(f"  sample kernel: name={k.get('name')!r} dur={k.get('dur')} args={k.get('args')}")
    return len(kernels)


def main():
    print("=== Profiler Foundation Verification Gate ===")
    prof = run_traced_ops()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        trace_path = f.name
    prof.export_chrome_trace(trace_path)
    keep = Path(trace_path)
    if len(sys.argv) > 1:
        keep = Path(sys.argv[1])
        keep.write_bytes(Path(trace_path).read_bytes())
        print(f"(trace copied to {keep})")

    try:
        with open(trace_path) as f:
            trace = json.load(f)

        print("\n--- Metric 1: flow arrows ---")
        flow_count = verify_flow_arrows(trace)
        print("\n--- Metric 2: device time attribution ---")
        device_time = verify_device_time_attribution(prof)
        print("\n--- Metric 3: runtime events ---")
        runtime_cat = verify_runtime_category(
            trace, expected_cats={"privateuse1_runtime", "cuda_runtime"}
        )
        print("\n--- Context ---")
        kernel_event_summary(trace)

        print("\n=== RESULT ===")
        m1 = flow_count > 0
        m2 = device_time is not None and device_time > 0
        m3 = runtime_cat is not None
        print(f"  metric1 flow arrows        : {'PASS' if m1 else 'FAIL'} ({flow_count})")
        print(f"  metric2 aten::mm dev time   : {'PASS' if m2 else 'FAIL'} ({device_time}us)")
        print(f"  metric3 runtime category    : {'PASS' if m3 else 'FAIL'} ({runtime_cat})")
        if m1 and m2 and m3:
            print("ALL METRICS PASSED")
            return 0
        print("SOME METRICS FAILED")
        return 1
    finally:
        if keep != Path(trace_path):
            Path(trace_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
