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

"""Unit tests for the PrivateUse1 "flagos" profiler support work.

Run (must go through the CUDA libtorch preload wrapper):
    FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
      bash scripts/with_cuda_libtorch.sh python -m pytest \
      tests/unit/test_profiler_privateuse1.py -v
"""

import os, sys
import torch
import torch_fl  # noqa: F401


def test_guard_stream_is_real_not_synthetic():
    """guard 返回的 current stream id 应是真实 CUDA stream id，而不是恒为 0
    的合成流。

    NOTE (deviation from the plan's literal snippet): the plan's draft used
    ``torch.flagos.current_stream()`` / ``torch.flagos.Stream()`` /
    ``torch.flagos.stream(s)``. Those are pure-Python proxies in
    torch_fl/flagos/__init__.py aimed at FSDP compatibility -- they forward
    to ``torch.cuda.*``, which under the CPU-only torch wheel is either a
    dummy ``_StreamShim`` (unconditionally stream_id/cuda_stream=0, wired up
    by torch_fl.accelerator.cuda._cuda_compat for triton) or, for
    ``torch.cuda.Stream``, raises "Tried to instantiate dummy base class
    Stream" (the CPU wheel's native _CudaStreamBase is an unconditional
    stub). Neither path reaches c10::flagos::GuardImpl at all, so that
    snippet can't actually exercise this task's change and was confirmed to
    fail for reasons unrelated to guard.h.

    torch's device-agnostic ``torch.accelerator`` / ``torch.Stream(device=...)``
    API, by contrast, does dispatch through the registered PrivateUse1
    GuardImpl (getStream/getNewStream/exchangeStream) -- the same methods
    Step 3 modifies -- so it is used here instead, keeping the original
    intent: two distinct streams must report distinct, non-synthetic ids.
    """
    dev = torch.device("flagos", 0)
    s0 = torch.accelerator.current_stream(dev)
    # 合成流恒为 0；用两条不同流断言 id 不同来证明不是写死的 0
    s1 = torch.Stream(device=dev)
    s2 = torch.Stream(device=dev)
    assert s1.stream_id != s2.stream_id

    with s2:
        cur = torch.accelerator.current_stream(dev)
        assert cur.stream_id == s2.stream_id

    # Restored after the context manager exits.
    assert torch.accelerator.current_stream(dev).stream_id == s0.stream_id
    assert s0.stream_id != s1.stream_id or s0.stream_id != s2.stream_id
