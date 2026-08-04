"""
Verification gate for profiler parity design foundation (Task 1, one-shot).

Tests 3 claims:
1. Chrome trace contains *renderable* flow arrows (ac2g, paired s<->f, count > 0)
2. key_averages() shows aten::mm with self_device_time_total > 0, and that value
   equals the summed duration of the device events (sgemm kernels + the cuBLAS
   workspace memsets) linked to aten::mm in the same trace
3. Runtime events appear with a recognizable runtime category, and their names
   actually come from the cbid mapping rather than a hardcoded constant

Plus a capture-window containment check: no runtime/kernel/memcpy event may fall
outside the CPU op window.

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
    """Metric 1: count of *renderable* (paired) ac2g flow arrows.

    A bare count of ac2g entries is NOT a valid check: an earlier iteration of
    this work emitted 203 unpaired 'f' halves, which no viewer can draw, and a
    `count > 0` assertion passed anyway. An arrow is renderable only when an 's'
    half and an 'f' half share a flow id, so we count matched ids.
    """
    top_level = [e for e in trace.get("flowEvents", []) if e.get("cat") == "ac2g"]
    inline = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("cat") == "ac2g" and e.get("ph") in ("s", "f")
    ]
    flows = top_level + inline
    phases = Counter(e.get("ph") for e in flows)
    start_ids = {e["id"] for e in flows if e.get("ph") == "s"}
    finish_ids = {e["id"] for e in flows if e.get("ph") == "f"}
    paired_ids = start_ids & finish_ids

    print(f"ac2g entries: {len(flows)} "
          f"(top-level flowEvents={len(top_level)}, inline ph=s/f={len(inline)})")
    print(f"  ph counts: {dict(phases)}")
    print(f"  ids: s={len(start_ids)} f={len(finish_ids)} paired={len(paired_ids)}")

    dangling_s = start_ids - finish_ids
    dangling_f = finish_ids - start_ids
    if dangling_s or dangling_f:
        print(f"  DANGLING: {len(dangling_s)} 's' without 'f', "
              f"{len(dangling_f)} 'f' without 's'")

    ok = (
        len(paired_ids) > 0
        and phases.get("s", 0) == phases.get("f", 0)
        and start_ids == finish_ids
    )
    print(f"Flow arrows (paired, renderable): {len(paired_ids)}")
    return len(paired_ids), ok


def aten_mm_device_time(trace):
    """Ground truth for metric 2: total device time aten::mm actually owns.

    aten::mm's attributed time must equal the sum of the device events it
    actually launched. Without this cross-check, `device_time > 0` also passes
    when a stray unrelated kernel's time is misattributed to aten::mm.

    ORIGINALLY this summed only the sgemm kernel events, because those were the
    only device activities collected. Task 3 added MEMSET collection, and cuBLAS
    zeroes a 512B workspace per gemm -- so aten::mm now legitimately owns 5
    memsets (~2us each) in addition to its 5 sgemm kernels, and a sgemm-only
    ground truth under-counts by exactly that amount.
    Verified by same-binary A/B: with MEMSET disabled at the ActivityEnable call
    and nothing else changed, attribution returns to the sgemm sum exactly
    (816.059us vs 816.059us, delta 0.000us).

    So the truth is now "every device event linked to aten::mm", which is
    STRICTER than the old form, not looser: it still catches a missing/extra
    kernel (~163us), and additionally catches a memset misattributed to the
    wrong op. The sgemm sub-total is still asserted separately below so that
    losing kernel collection entirely cannot pass.
    """
    events = trace.get("traceEvents", [])
    # cpu_op "External id" -> op name, to attribute device events to their op.
    op_by_ext = {}
    for e in events:
        if e.get("cat") == "cpu_op":
            ext = (e.get("args") or {}).get("External id")
            if ext is not None:
                op_by_ext[ext] = e.get("name")

    def linked_to_mm(e):
        ext = (e.get("args") or {}).get("External id")
        return op_by_ext.get(ext) == "aten::mm"

    kernels = [
        e for e in events
        if e.get("cat") == "kernel" and "sgemm" in e.get("name", "").lower()
    ]
    memsets = [e for e in events if e.get("cat") == "gpu_memset" and linked_to_mm(e)]

    kernel_total = sum(e.get("dur", 0) for e in kernels)
    memset_total = sum(e.get("dur", 0) for e in memsets)
    print(f"Ground truth: {len(kernels)} sgemm kernel events, sum dur={kernel_total}us")
    print(f"              + {len(memsets)} memset events linked to aten::mm, "
          f"sum dur={memset_total}us")
    print(f"              = {kernel_total + memset_total}us total")
    return kernel_total + memset_total, len(kernels)


def verify_device_time_attribution(prof, trace):
    """Metric 2: aten::mm self_device_time_total, cross-checked against the
    device events in the same trace."""
    key_avg = prof.key_averages()
    mm_events = [e for e in key_avg if "mm" in e.key.lower()]
    for evt in mm_events:
        print(f"  {evt.key}: self_device_time_total={evt.self_device_time_total}us "
              f"device_time_total={evt.device_time_total}us count={evt.count}")

    attributed = None
    for evt in mm_events:
        if "aten::mm" in evt.key:
            attributed = evt.self_device_time_total
            break
    if attributed is None:
        print("aten::mm NOT FOUND in key_averages()")
        return None, False

    print(f"aten::mm self_device_time_total={attributed}us")
    truth, n_kernels = aten_mm_device_time(trace)
    if n_kernels == 0:
        print("  FAIL: no sgemm kernel events to cross-check against")
        return attributed, False

    # Tolerance is generous in absolute terms but far tighter than the failure
    # modes it guards: one extra/missing kernel would shift this by ~163us.
    delta = abs(attributed - truth)
    ok = attributed > 0 and delta <= max(1.0, 0.01 * truth)
    print(f"  cross-check: attributed={attributed}us vs device event sum={truth}us "
          f"delta={delta:.3f}us -> {'MATCH' if ok else 'MISMATCH'}")
    return attributed, ok


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


def verify_runtime_names_from_cbid(trace, expected_cats):
    """Metric 3b: runtime event names must actually come from the cbid mapping.

    INTERIM GUARD (Task 3 review). The cbid->name table in cupti_shim.h is this
    layer's only defence against a class of bug whose signature is "a plausible
    but wrong label rather than an error": before it existed, EVERY runtime
    record was hardcoded to "cudaLaunchKernel", and a 21.9ms blocking
    synchronize was reported as a kernel launch. Nothing in the build re-checks
    that table, so a regression in Task 4/5 would be silent.

    The cheap invariant that catches exactly the old hardcode: a real workload
    calls more than one runtime API, so at least one runtime event must carry a
    name that is neither the generic `default:` fallback ("cudaRuntime") nor
    "cudaLaunchKernel". If the mapping is bypassed or reverted to a constant,
    every name collapses to one of those two and this fails.

    A proper table-driven test belongs in Task 6's integration test (the repo
    has no C++ test harness today); this only has to hold the line until then.
    """
    runtime_events = [
        e for e in trace.get("traceEvents", []) if e.get("cat") in expected_cats
    ]
    names = Counter(e.get("name") for e in runtime_events)
    generic = {"cudaRuntime", "cudaLaunchKernel"}
    specific = {n: c for n, c in names.items() if n not in generic}
    print(f"Runtime event names: {dict(names)}")
    if not runtime_events:
        print("  FAIL: no runtime events at all to check names on")
        return False
    print(f"  names beyond {sorted(generic)}: {dict(specific) or 'NONE'}")
    ok = bool(specific)
    print(f"  cbid->name mapping in effect: {'OK' if ok else 'NOT IN EFFECT'}")
    if not ok:
        print("    Every runtime event is 'cudaRuntime' or 'cudaLaunchKernel'.")
        print("    Either the cbid table regressed to a hardcoded name, or the")
        print("    runtime name is no longer sourced from the record's cbid.")
    return ok


def kernel_event_summary(trace):
    """Context: how many device kernels landed, and do they carry a correlation?"""
    kernels = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]
    with_corr = [
        k for k in kernels if (k.get("args") or {}).get("correlation") not in (None, 0)
    ]
    print(f"Kernel events: {len(kernels)} (with non-zero correlation arg: {len(with_corr)})")
    return len(kernels)


def window_containment(trace):
    """Finding 5 check: no device/runtime event may sit outside the capture window.

    The authoritative window is the "PyTorch Profiler" trace span (cat "Trace"),
    which is what kineto passes to processTrace as [startTime, endTime]. The
    cpu_op extent is a poor proxy: it ends at the last op, while the real window
    runs to profiler stop, so a legitimate trailing launch looks like a
    violation.
    """
    ev = trace.get("traceEvents", [])
    span = [e for e in ev if e.get("cat") == "Trace"]
    if not span:
        print("no Trace span event; cannot check window containment")
        return True
    s = max(span, key=lambda e: e.get("dur", 0))
    lo = s["ts"]
    hi = lo + s.get("dur", 0)
    width = hi - lo

    tracked = [
        e for e in ev
        if e.get("cat") in ("privateuse1_runtime", "kernel", "gpu_memcpy")
    ]
    before = [e for e in tracked if e["ts"] + e.get("dur", 0) < lo]
    after = [e for e in tracked if e["ts"] > hi]
    longest = max((e.get("dur", 0) for e in tracked), default=0)

    print(f"capture window ({s['name']}): span={width:.1f}us")
    print(f"  events entirely before window: {len(before)}/{len(tracked)}")
    print(f"  events entirely after window : {len(after)}/{len(tracked)}")
    print(f"  longest tracked event: {longest:.1f}us (window span {width:.1f}us)")
    for e in (before + after)[:5]:
        print(f"    OUT: cat={e['cat']} name={e['name'][:50]!r} dur={e.get('dur')}")
    ok = not before and not after and longest <= width
    print(f"  window containment: {'OK' if ok else 'VIOLATED'}")
    return ok


def verify_kernel_dominates(trace):
    """Guard against the ground truth in metric 2 degenerating.

    Metric 2 now sums kernels + memsets. If kernel collection silently broke and
    only memsets survived, both sides of that comparison would shrink together
    and still "MATCH". Assert separately that the sgemm kernels are present and
    account for the bulk of aten::mm's device time (they are ~163us each vs
    ~2us per memset, so >90% is a wide margin around the real ~98.9%).
    """
    events = trace.get("traceEvents", [])
    op_by_ext = {
        (e.get("args") or {}).get("External id"): e.get("name")
        for e in events if e.get("cat") == "cpu_op"
    }
    sgemm = [e for e in events
             if e.get("cat") == "kernel" and "sgemm" in e.get("name", "").lower()]
    memsets = [e for e in events if e.get("cat") == "gpu_memset"
               and op_by_ext.get((e.get("args") or {}).get("External id")) == "aten::mm"]
    k = sum(e.get("dur", 0) for e in sgemm)
    m = sum(e.get("dur", 0) for e in memsets)
    share = k / (k + m) if (k + m) else 0.0
    print(f"sgemm kernels: {len(sgemm)} ({k:.1f}us), aten::mm memsets: "
          f"{len(memsets)} ({m:.1f}us) -> kernels are {share:.1%} of the total")
    ok = len(sgemm) >= 5 and share > 0.90
    print(f"  kernel dominance: {'OK' if ok else 'DEGENERATE'}")
    return ok


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
        flow_count, m1 = verify_flow_arrows(trace)
        print("\n--- Metric 2: device time attribution ---")
        device_time, m2 = verify_device_time_attribution(prof, trace)
        m2b = verify_kernel_dominates(trace)
        m2 = m2 and m2b
        print("\n--- Metric 3: runtime events ---")
        runtime_cat = verify_runtime_category(
            trace, expected_cats={"privateuse1_runtime", "cuda_runtime"}
        )
        m3 = runtime_cat is not None
        m3b = verify_runtime_names_from_cbid(
            trace, expected_cats={"privateuse1_runtime", "cuda_runtime"}
        )
        print("\n--- Context ---")
        kernel_event_summary(trace)
        print("\n--- Capture-window containment (finding 5) ---")
        m4 = window_containment(trace)

        print("\n=== RESULT ===")
        print(f"  metric1 paired flow arrows  : {'PASS' if m1 else 'FAIL'} ({flow_count})")
        print(f"  metric2 aten::mm dev time   : {'PASS' if m2 else 'FAIL'} ({device_time}us)")
        print(f"  metric3 runtime category    : {'PASS' if m3 else 'FAIL'} ({runtime_cat})")
        print(f"  metric3b cbid->name mapping : {'PASS' if m3b else 'FAIL'}")
        print(f"  window containment          : {'PASS' if m4 else 'FAIL'}")
        if m1 and m2 and m3 and m3b and m4:
            print("ALL METRICS PASSED")
            return 0
        print("SOME METRICS FAILED")
        return 1
    finally:
        if keep != Path(trace_path):
            Path(trace_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
