# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MUSA MUPTI activity capture on a physical Moore Threads device."""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.musa


def _require_musa():
    if os.environ.get("ACCELERATOR", "").lower() != "musa":
        pytest.skip("requires an ACCELERATOR=musa torch-fl build")
    if not hasattr(torch, "flagos") or not torch.flagos.is_available():
        pytest.skip("requires an available MUSA device")
    if not Path("/usr/local/musa/lib/libmupti.so").exists():
        pytest.skip("requires the MUSA MUPTI runtime")


def _trace_events(trace, category):
    return [
        event
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") == category
    ]


def test_musa_mupti_captures_device_activity(tmp_path):
    """MUPTI produces valid, positive-duration MUSA kernel and runtime events."""
    import torch_fl  # noqa: F401 - registers the flagos backend and profiler

    _require_musa()
    device = torch.device("flagos", 0)
    x = torch.randn(64, 64, device=device)
    y = torch.randn(64, 64, device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        result = (x @ y).relu()
        result.sum().item()

    trace_path = tmp_path / "musa-mupti-trace.json"
    prof.export_chrome_trace(str(trace_path))
    with trace_path.open() as stream:
        trace = json.load(stream)

    kernels = _trace_events(trace, "kernel")
    runtimes = _trace_events(trace, "privateuse1_runtime")
    assert kernels, "MUPTI produced no MUSA kernel activities"
    assert runtimes, "MUPTI produced no MUSA runtime activities"
    assert all(event.get("name") for event in kernels)
    assert all(event.get("dur", 0) > 0 for event in kernels)
    assert all(event.get("dur", 0) > 0 for event in runtimes)
    assert all((event.get("args") or {}).get("device") is not None for event in kernels)
    assert all((event.get("args") or {}).get("stream") is not None for event in kernels)

    # The CPU-only PyTorch wheel may not provide kineto's PrivateUse1 resolver;
    # device capture remains a required result, while linkage is asserted only
    # when the wheel emits external ids for the captured kernels.
    linked = [
        event
        for event in kernels
        if (event.get("args") or {}).get("External id") is not None
    ]
    if linked:
        assert any(
            event.get("cat") == "cpu_op"
            and (event.get("args") or {}).get("External id") is not None
            for event in trace.get("traceEvents", [])
        )
