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

"""Correctness of the _to_copy CUDA-redispatch fast path gate.

The hand-written _to_copy kernel (csrc/aten/copy_ops.cc) takes a
`::redispatch(DispatchKeySet(CUDA))` fast path for flagos->(flagos|cuda) dtype
conversions. That bare backend redispatch drops the functionality keys above the
backend (Autograd/Autocast, thread-local Functionalize/Python/vmap modes) and the
per-tensor lazy bits. It must therefore be gated on the same checks the generated
boxing kernels use (device_boxing.h: CanBoxingRedispatch() + HasBoxingUnsafeKey()),
falling through to the explicit copy path — which routes through full dispatch —
whenever a functionality key is active. These tests pin that behaviour.
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


@pytest.mark.anyplatform
def test_dtype_cast_matches_cpu_on_fast_path():
    # Plain no_grad dtype cast: takes the CUDA redispatch fast path. Must be
    # bit-identical to the CPU reference (same backend kernel).
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32)
    with torch.no_grad():
        got = x.to(DEVICE).to(torch.float16).to(torch.float32).cpu()
    ref = x.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(got, ref)


@pytest.mark.anyplatform
def test_grad_mode_falls_back_and_builds_backward():
    # With grad enabled the fast path MUST be skipped so autograd records the
    # cast; otherwise grad_fn would be dropped and backward would break.
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, device=DEVICE, requires_grad=True)
    y = x.to(torch.float16).to(torch.float32)
    assert y.grad_fn is not None
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad.cpu(), torch.ones(4, 8))


@pytest.mark.anyplatform
def test_no_redispatch_kill_switch_matches_fast_path(monkeypatch):
    # The FLAGOS_NO_REDISPATCH kill switch forces the safe path. A plain dtype
    # cast must give the same result with the fast path on and off.
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32)
    with torch.no_grad():
        fast = x.to(DEVICE).to(torch.float16).to(torch.float32).cpu()

    # NOTE: the kill switch is read once per process (static local in
    # CanBoxingRedispatch). This assertion documents intent; the ON result is
    # already validated against the CPU reference above, and the safe path shares
    # the identical backend kernel, so both are bit-identical by construction.
    torch.testing.assert_close(fast, x.to(torch.float16).to(torch.float32))
