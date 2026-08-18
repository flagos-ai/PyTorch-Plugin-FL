# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small structural profiler gate for Ascend MSPTI."""

import ctypes
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.ascend


@pytest.fixture(scope="module")
def traced_profiles():
    """Run two sessions so stale MSPTI state cannot make one pass."""
    import torch_fl  # noqa: F401

    if os.environ.get("ACCELERATOR", "").lower() != "ascend":
        pytest.skip("requires an Ascend torch-fl build")
    if not any(name.startswith("davinci") for name in os.listdir("/dev")):
        pytest.skip("requires Ascend device nodes")

    acl = ctypes.CDLL(None)
    acl.aclrtMemset.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_size_t,
    ]
    acl.aclrtMemset.restype = ctypes.c_int
    acl.aclrtMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    acl.aclrtMemsetAsync.restype = ctypes.c_int

    results = []
    for _ in range(2):
        x = torch.randn(16, 16, device="flagos:0")
        y = torch.randn(16, 16, device="flagos:0")
        fill = torch.empty(1024 * 1024, dtype=torch.uint8, device="flagos:0")
        host = torch.randn(32)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ]
        ) as prof:
            z = (x @ y).relu()
            device_copy = host.to("flagos:0")
            device_copy.cpu()
            fill_ptr = ctypes.c_void_p(fill.data_ptr())
            assert acl.aclrtMemset(fill_ptr, fill.numel(), 17, fill.numel()) == 0
            stream = torch.accelerator.current_stream(torch.device("flagos", 0))
            assert (
                acl.aclrtMemsetAsync(
                    fill_ptr,
                    fill.numel(),
                    23,
                    fill.numel(),
                    ctypes.c_void_p(stream.stream_id),
                )
                == 0
            )
            z.sum().item()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as trace_file:
            path = Path(trace_file.name)
        try:
            prof.export_chrome_trace(str(path))
            with path.open() as stream:
                trace = json.load(stream)
        finally:
            path.unlink(missing_ok=True)
        matmul_device_time = next(
            (
                item.device_time_total
                for item in prof.key_averages()
                if item.key in ("aten::matmul", "aten::mm")
            ),
            None,
        )
        results.append((matmul_device_time, trace))
    return results


def _events(trace, category):
    return [
        event
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") == category
    ]


def test_ascend_trace_has_real_device_activity(traced_profiles):
    """Each session contains named, positive-duration MSPTI device work."""
    for _, trace in traced_profiles:
        kernels = _events(trace, "kernel")
        runtimes = _events(trace, "privateuse1_runtime")
        assert kernels, "MSPTI did not produce Ascend kernel activities"
        assert runtimes, "MSPTI did not produce Ascend runtime activities"
        assert all(
            event.get("name") not in (None, "", "AscendKernel") for event in kernels
        )
        assert all(event.get("dur", 0) > 0 for event in kernels)
        assert all(event.get("dur", 0) > 0 for event in runtimes)
        assert all(
            (event.get("args") or {}).get("device") is not None for event in kernels
        )
        assert all(
            (event.get("args") or {}).get("stream") is not None for event in kernels
        )


def test_ascend_trace_has_memory_activity_when_preloaded(traced_profiles):
    """CANN memory activities require process-start LD_PRELOAD."""
    if "mspti" not in os.environ.get("LD_PRELOAD", "").lower():
        pytest.skip("CANN requires libmspti.so in LD_PRELOAD for memory interception")
    for _, trace in traced_profiles:
        copies = _events(trace, "gpu_memcpy")
        assert copies, "MSPTI preload produced no Ascend memcpy activities"
        assert all(event.get("dur", 0) > 0 for event in copies)
        assert all((event.get("args") or {}).get("bytes", 0) > 0 for event in copies)


def test_ascend_trace_has_memset_activity_when_preloaded(traced_profiles):
    """CANN emits device memset records through process-start interposition."""
    if "mspti" not in os.environ.get("LD_PRELOAD", "").lower():
        pytest.skip("requires process-start MSPTI interception")
    for _, trace in traced_profiles:
        memsets = _events(trace, "gpu_memset")
        assert len(memsets) >= 2, "MSPTI produced no sync/async memset activities"
        assert all(event.get("dur", 0) > 0 for event in memsets)
        assert all(
            (event.get("args") or {}).get("bytes") == 1024 * 1024 for event in memsets
        )
        assert {
            int((event.get("args") or {}).get("async", -1)) for event in memsets
        } >= {
            0,
            1,
        }


def test_ascend_trace_has_paired_flows(traced_profiles):
    """Runtime/device correlation IDs form renderable ac2g arrows."""
    for _, trace in traced_profiles:
        flows = [
            event
            for event in trace.get("traceEvents", [])
            if event.get("cat") == "ac2g" and event.get("ph") in ("s", "f")
        ]
        assert flows, "Ascend trace has no CPU-to-device flow arrows"
        starts = {event["id"] for event in flows if event["ph"] == "s"}
        finishes = {event["id"] for event in flows if event["ph"] == "f"}
        assert starts == finishes


def test_ascend_trace_has_cpu_linkage(traced_profiles):
    """Matmul device time reconciles with its linked kernel activity."""
    for matmul_device_time, trace in traced_profiles:
        assert any(event.get("cat") == "cpu_op" for event in trace["traceEvents"])
        op_names = {
            (event.get("args") or {}).get("External id"): event.get("name")
            for event in _events(trace, "cpu_op")
            if (event.get("args") or {}).get("External id") is not None
        }
        linked_kernels = [
            event
            for event in _events(trace, "kernel")
            if (event.get("args") or {}).get("External id") in op_names
        ]
        assert linked_kernels, "MSPTI external correlation did not link kernels"
        assert matmul_device_time is not None, (
            "matmul aggregation is absent from profiler results"
        )
        assert matmul_device_time > 0
        matmul_kernels = [
            event
            for event in _events(trace, "kernel")
            if event.get("name", "").startswith("MatMul")
        ]
        assert matmul_kernels, "The Ascend matmul kernel is absent from the trace"
        truth = sum(event.get("dur", 0) for event in matmul_kernels)
        assert abs(matmul_device_time - truth) <= max(1.0, truth * 0.01)
