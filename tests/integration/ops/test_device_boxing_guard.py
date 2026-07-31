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

"""Generic lifetime and restoration coverage for CUDA metadata boxing guards."""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


@pytest.mark.anyplatform
def test_duplicate_tensor_alias_is_restored_after_boxing():
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensor = values.to(DEVICE)

    actual = torch.add(tensor, tensor)

    assert tensor.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), values + values)
    torch.testing.assert_close(tensor.cpu(), values)


@pytest.mark.anyplatform
def test_genuine_cpu_scalar_tensor_is_not_reboxed():
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    tensor = values.to(DEVICE)
    cpu_scalar = torch.tensor(2.5)

    actual = torch.mul(tensor, cpu_scalar)

    assert tensor.device.type == "flagos"
    assert cpu_scalar.device.type == "cpu"
    torch.testing.assert_close(actual.cpu(), values * cpu_scalar)


@pytest.mark.anyplatform
def test_boxed_inputs_are_restored_when_native_kernel_throws():
    left_values = torch.randn(2, 3)
    right_values = torch.randn(4, 5)
    left = left_values.to(DEVICE)
    right = right_values.to(DEVICE)

    with pytest.raises(RuntimeError):
        torch.mm(left, right)

    assert left.device.type == "flagos"
    assert right.device.type == "flagos"
    torch.testing.assert_close(left.cpu(), left_values)
    torch.testing.assert_close(right.cpu(), right_values)


@pytest.mark.anyplatform
def test_tensorlist_longer_than_inline_capacity_is_restored():
    values = [torch.full((2, 3), float(i)) for i in range(8)]
    tensors = [value.to(DEVICE) for value in values]

    actual = torch.cat(tensors, dim=0)

    assert all(tensor.device.type == "flagos" for tensor in tensors)
    torch.testing.assert_close(actual.cpu(), torch.cat(values, dim=0))
