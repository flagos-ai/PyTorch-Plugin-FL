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

"""Generic correctness and aliasing coverage for hand-written metadata ops."""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _assert_same_view(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.device.type == "flagos"
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "operation",
    [
        lambda tensor: tensor.transpose(0, -1),
        lambda tensor: tensor.permute(2, 0, 1),
        lambda tensor: tensor.select(-1, 1),
        lambda tensor: torch.ops.aten.slice.Tensor(tensor, -1, 1, 4, 2),
        lambda tensor: tensor.unsqueeze(-2),
    ],
    ids=["transpose", "permute", "select", "slice", "unsqueeze"],
)
def test_metadata_views_match_cpu_and_alias_storage(operation):
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    tensor = values.to(DEVICE)

    actual = operation(tensor)
    expected = operation(values)

    _assert_same_view(actual, expected)
    assert actual.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


@pytest.mark.anyplatform
def test_squeeze_overloads_match_cpu_and_alias_storage():
    values = torch.arange(6, dtype=torch.float32).reshape(1, 2, 1, 3)
    tensor = values.to(DEVICE)

    for actual, expected in (
        (torch.squeeze(tensor), torch.squeeze(values)),
        (torch.squeeze(tensor, 2), torch.squeeze(values, 2)),
        (torch.squeeze(tensor, -1), torch.squeeze(values, -1)),
    ):
        _assert_same_view(actual, expected)
        assert actual.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


@pytest.mark.anyplatform
def test_unsafe_view_matches_cpu_and_aliases_storage():
    values = torch.arange(24, dtype=torch.float32)
    tensor = values.to(DEVICE)

    actual = torch.ops.aten._unsafe_view.default(tensor, (4, 6))
    expected = torch.ops.aten._unsafe_view.default(values, (4, 6))

    _assert_same_view(actual, expected)
    assert actual.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


@pytest.mark.anyplatform
def test_detach_preserves_device_data_and_storage_alias():
    values = torch.randn(3, 5, dtype=torch.float32)
    tensor = values.to(DEVICE).requires_grad_(True)

    detached = tensor.detach()

    assert detached.device.type == "flagos"
    assert not detached.requires_grad
    assert detached.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    torch.testing.assert_close(detached.cpu(), values)


@pytest.mark.anyplatform
def test_metadata_views_preserve_empty_dimensions():
    values = torch.empty(2, 0, 3)
    tensor = values.to(DEVICE)

    actual = tensor.permute(1, 2, 0).select(-1, 1).unsqueeze(0)
    expected = values.permute(1, 2, 0).select(-1, 1).unsqueeze(0)

    _assert_same_view(actual, expected)
