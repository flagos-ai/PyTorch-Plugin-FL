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

"""TOPSPTI activity capture on a physical Enflame GCU device."""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gcu


def _require_gcu():
    if os.environ.get("ACCELERATOR", "").lower() != "gcu":
        pytest.skip("requires an ACCELERATOR=gcu torch-fl build")
    if not hasattr(torch, "flagos") or not torch.flagos.is_available():
        pytest.skip("requires an available GCU device")
    sdk_root = Path(os.environ.get("TOPS_HOME", "/opt/tops"))
    if not (sdk_root / "extras/TOPSPTI/lib64/libtopspti.so").exists():
        pytest.skip("requires the GCU TOPSPTI runtime")


def _trace_events(trace, category):
    return [
        event
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") == category
    ]


def test_gcu_topspti_captures_device_activity(tmp_path):
    """TOPSPTI produces valid, positive-duration GCU kernel/runtime events."""
    import torch_fl  # noqa: F401 - registers the flagos backend and profiler

    _require_gcu()
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

    trace_path = tmp_path / "gcu-topspti-trace.json"
    prof.export_chrome_trace(str(trace_path))
    with trace_path.open() as stream:
        trace = json.load(stream)

    kernels = _trace_events(trace, "kernel")
    runtimes = _trace_events(trace, "privateuse1_runtime")
    assert kernels, "TOPSPTI produced no GCU kernel activities"
    assert runtimes, "TOPSPTI produced no GCU runtime activities"
    assert all(event.get("name") for event in kernels)
    assert all(event.get("dur", 0) > 0 for event in kernels)
    assert all(event.get("dur", 0) > 0 for event in runtimes)
    assert all((event.get("args") or {}).get("device") is not None for event in kernels)
    assert all((event.get("args") or {}).get("stream") is not None for event in kernels)
