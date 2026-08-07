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

"""Regression tests for the two gaps that only bite *multi-kernel* graphs.

``test_compile.py`` compiles single-``Linear`` models, which need neither
Triton autotuning nor parallel compilation, so both of the failures guarded
here slipped through it:

1. **Autotuning needs a constructible event.** Inductor's
   ``InductorBenchmarker.get_event_pairs`` builds ``torch.cuda.Event(
   enable_timing=True)``. On the CPU-torch wheel that class derives from a
   dummy type and raises on construction, so any graph with more than one
   Triton config to choose between died with "Tried to instantiate dummy base
   class Event".

2. **Compile workers need torch_fl.** Inductor farms kernel compilation out to
   subprocesses that import only ``torch`` and ``triton``. Without ``torch_fl``
   the CUDA driver never registers, and triton reports "Could not find an
   active GPU backend".

Both are properties of the *shape* of the graph rather than of any op, so the
models below are chosen to force multiple kernels: stacked linears (autotuning)
plus normalizations and reductions (enough kernels to go parallel).
"""

import pytest
import torch
import torch_fl


@pytest.fixture
def device():
    if torch_fl.flagos.device_count() == 0:
        pytest.skip("No flagos devices available")
    return "flagos:0"


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Each test compiles from scratch; a cached graph would prove nothing."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _assert_matches_eager(model, *args, rtol=1e-3, atol=1e-3):
    """Compile ``model``, and check it lands on flagos with eager's numbers."""
    expected = model(*args)
    actual = torch.compile(model, backend="flagos")(*args)
    assert actual.device.type in ("privateuseone", "flagos"), (
        f"output landed on {actual.device}, expected flagos"
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return actual


def test_event_is_constructible(device):
    """The autotuner's exact call. Guards gap 1 at its narrowest point."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    x = torch.randn(2048, 2048, device=device)
    start.record()
    for _ in range(10):
        x = x * 1.001
    end.record()
    end.synchronize()

    # Real device timing, not a host stand-in: work this size is never free,
    # and never takes a minute.
    elapsed = start.elapsed_time(end)
    assert 0.0 < elapsed < 60_000.0, f"implausible elapsed_time: {elapsed}"


def test_flagos_event_is_constructible(device):
    """``flagos.Event`` is what ``torch.cuda.Event`` is routed to."""
    event = torch_fl.flagos.Event(enable_timing=True)
    event.record()
    event.synchronize()
    assert event.query() is True


def test_compile_stacked_linears(device):
    """Two matmuls give the autotuner configs to pick between (gap 1)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 512),
    ).to(device)
    _assert_matches_eager(model, torch.randn(64, 512, device=device))


def test_compile_normalizations(device):
    """LayerNorms emit enough kernels to reach parallel compilation (gap 2)."""
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(512),
        torch.nn.Linear(512, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 512),
        torch.nn.LayerNorm(512),
    ).to(device)
    _assert_matches_eager(model, torch.randn(64, 512, device=device))


def test_compile_reductions(device):
    """Softmax/sum/mean are reduction kernels -- a different codegen path."""

    def f(x):
        return (x * 2.0).softmax(-1).sum(-1) + x.mean(-1)

    _assert_matches_eager(f, torch.randn(256, 512, device=device))


def test_compile_backward_multikernel(device):
    """Backward doubles the kernel count, so it hits both gaps hardest."""
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 512),
    ).to(device)
    x = torch.randn(64, 512, device=device, requires_grad=True)

    torch.compile(model, backend="flagos")(x).sum().backward()

    assert x.grad is not None, "no gradient reached the input"
    assert x.grad.device.type in ("privateuseone", "flagos")
    assert torch.isfinite(x.grad).all(), "gradient has non-finite entries"


def test_compile_dynamic_shapes(device):
    """One compile serving several batch sizes still has to autotune."""
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
    ).to(device)
    compiled = torch.compile(model, backend="flagos", dynamic=True)

    for batch in (16, 32, 64):
        x = torch.randn(batch, 512, device=device)
        actual = compiled(x)
        assert actual.shape == (batch, 512)
        torch.testing.assert_close(actual, model(x), rtol=1e-3, atol=1e-3)


def test_compile_max_autotune(device):
    """``max-autotune`` benchmarks candidates, so it cannot skip the events."""
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 256),
    ).to(device)
    x = torch.randn(64, 256, device=device)

    expected = model(x)
    actual = torch.compile(model, backend="flagos", mode="max-autotune")(x)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
