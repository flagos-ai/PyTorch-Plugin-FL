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

"""GroupNorm correctness for standard and non-standard input layouts."""

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"
TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}


def _make_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        return tensor.contiguous()
    if layout == "channels_last":
        return tensor.to(memory_format=torch.channels_last)
    if layout == "strided":
        expanded = torch.empty(
            *tensor.shape[:-1],
            tensor.shape[-1] * 2,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        expanded[..., ::2].copy_(tensor)
        return expanded[..., ::2]
    raise AssertionError(f"unknown layout: {layout}")


@pytest.mark.parametrize(
    "shape,groups,layout",
    [
        ((2, 64, 8, 8), 8, "contiguous"),
        ((1, 64, 8, 8), 8, "channels_last"),
        ((2, 8, 5, 7), 1, "strided"),
        ((2, 8, 5, 7), 8, "strided"),
        ((2, 8, 17), 4, "contiguous"),
    ],
)
@pytest.mark.anyplatform
def test_group_norm_layouts_match_cpu(shape, groups, layout):
    generator = torch.Generator().manual_seed(20260728)
    input_cpu = torch.randn(shape, generator=generator)
    weight_cpu = torch.randn(shape[1], generator=generator)
    bias_cpu = torch.randn(shape[1], generator=generator)
    expected = F.group_norm(input_cpu, groups, weight_cpu, bias_cpu, 1e-5)

    input_device = _make_layout(input_cpu.to(DEVICE), layout)
    if layout != "contiguous":
        assert not input_device.is_contiguous()
    if layout == "channels_last":
        assert input_device.is_contiguous(memory_format=torch.channels_last)

    actual = F.group_norm(
        input_device,
        groups,
        weight_cpu.to(DEVICE),
        bias_cpu.to(DEVICE),
        1e-5,
    )

    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, **TOLERANCE)


@pytest.mark.anyplatform
def test_group_norm_channels_last_without_affine_matches_cpu():
    generator = torch.Generator().manual_seed(20260728)
    input_cpu = torch.randn(1, 64, 8, 8, generator=generator)
    input_device = input_cpu.to(DEVICE).to(memory_format=torch.channels_last)

    expected = F.group_norm(input_cpu, 8, None, None, 1e-5)
    actual = F.group_norm(input_device, 8, None, None, 1e-5)

    torch.testing.assert_close(actual.cpu(), expected, **TOLERANCE)


@pytest.mark.anyplatform
def test_group_norm_channels_last_backward_matches_cpu():
    generator = torch.Generator().manual_seed(20260728)
    input_values = torch.randn(2, 8, 5, 7, generator=generator)
    weight_values = torch.randn(8, generator=generator)
    bias_values = torch.randn(8, generator=generator)
    grad_values = torch.randn(input_values.shape, generator=generator)

    input_cpu = input_values.to(memory_format=torch.channels_last).requires_grad_(True)
    weight_cpu = weight_values.clone().requires_grad_(True)
    bias_cpu = bias_values.clone().requires_grad_(True)
    output_cpu = F.group_norm(input_cpu, 4, weight_cpu, bias_cpu, 1e-5)
    output_cpu.backward(grad_values)

    input_device = (
        input_values.to(DEVICE)
        .to(memory_format=torch.channels_last)
        .detach()
        .requires_grad_(True)
    )
    weight_device = weight_values.to(DEVICE).requires_grad_(True)
    bias_device = bias_values.to(DEVICE).requires_grad_(True)
    output_device = F.group_norm(input_device, 4, weight_device, bias_device, 1e-5)
    output_device.backward(grad_values.to(DEVICE))

    torch.testing.assert_close(output_device.cpu(), output_cpu.detach(), **TOLERANCE)
    torch.testing.assert_close(input_device.grad.cpu(), input_cpu.grad, **TOLERANCE)
    torch.testing.assert_close(weight_device.grad.cpu(), weight_cpu.grad, **TOLERANCE)
    torch.testing.assert_close(bias_device.grad.cpu(), bias_cpu.grad, **TOLERANCE)
