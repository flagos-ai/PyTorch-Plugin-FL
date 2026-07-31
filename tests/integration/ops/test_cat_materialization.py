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

"""Generic TensorList materialization coverage for cat and cat.out."""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


@pytest.mark.anyplatform
@pytest.mark.parametrize("count", [1, 4, 5, 16])
def test_cat_supports_lists_across_inline_capacity(count):
    cpu_inputs = [
        torch.full((2, 3), float(index), dtype=torch.float32)
        for index in range(count)
    ]
    device_inputs = [tensor.to(DEVICE) for tensor in cpu_inputs]

    actual = torch.cat(device_inputs, dim=0)
    expected = torch.cat(cpu_inputs, dim=0)

    assert actual.device.type == "flagos"
    assert all(tensor.device.type == "flagos" for tensor in device_inputs)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_cat_skips_only_legacy_empty_inputs():
    cpu_inputs = [
        torch.empty(0),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.empty(0),
    ]
    device_inputs = [tensor.to(DEVICE) for tensor in cpu_inputs]

    actual = torch.cat(device_inputs, dim=-2)
    expected = torch.cat(cpu_inputs, dim=-2)

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_cat_preserves_all_legacy_empty_semantics():
    cpu_inputs = [torch.empty(0), torch.empty(0), torch.empty(0)]
    device_inputs = [tensor.to(DEVICE) for tensor in cpu_inputs]

    actual = torch.cat(device_inputs)
    expected = torch.cat(cpu_inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device.type == "flagos"


@pytest.mark.anyplatform
def test_cat_does_not_skip_multidimensional_empty_inputs():
    cpu_inputs = [
        torch.empty(2, 0, 3),
        torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    ]
    device_inputs = [tensor.to(DEVICE) for tensor in cpu_inputs]

    actual = torch.cat(device_inputs, dim=1)
    expected = torch.cat(cpu_inputs, dim=1)

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_cat_out_uses_the_same_materialization_semantics():
    cpu_inputs = [
        torch.empty(0),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    ]
    device_inputs = [tensor.to(DEVICE) for tensor in cpu_inputs]
    expected = torch.cat(cpu_inputs, dim=-2)
    actual = torch.empty_like(expected, device=DEVICE)

    returned = torch.cat(device_inputs, dim=-2, out=actual)

    assert returned is actual
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected)
