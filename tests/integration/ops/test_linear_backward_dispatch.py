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

"""
linear_backward dispatch tests

nn.Linear on a >2-D input is the shape every transformer training step takes,
and it reaches a different code path than a 2-D one: autograd hands
linear_backward a grad_output that is routinely non-contiguous, which the
FlagGems op flattens with `.view(...).contiguous()` -- in that order, so the
view runs on the original layout and raises "view size is not compatible with
input tensor's size and stride". A 2-D check never sees it, so the ranks are
covered explicitly here.

Usage:
    pytest tests/integration/ops/test_linear_backward_dispatch.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# 2-D is the shape that already worked; 3-D is the transformer case that did
# not; 4-D confirms the flattening generalises past one batch dimension.
SHAPES = [(32, 64), (2, 16, 64), (2, 4, 8, 64)]


def _grads_for(shape, device):
    """Run one Linear forward+backward, returning (grad_in, grad_w, grad_b)."""
    torch.manual_seed(42)
    layer = nn.Linear(64, 128)
    layer = layer.to(device)
    x = torch.randn(*shape, device=device, requires_grad=True)
    layer(x).sum().backward()
    if device != "cpu":
        torch_fl.flagos.synchronize()
    return x.grad, layer.weight.grad, layer.bias.grad


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_linear_backward_matches_cpu(shape):
    """Gradients through nn.Linear match CPU autograd at every rank."""
    dev_in, dev_w, dev_b = _grads_for(shape, DEVICE)
    cpu_in, cpu_w, cpu_b = _grads_for(shape, "cpu")

    # grad_weight/grad_bias accumulate over the flattened batch, so they carry
    # more summation error than grad_input and get a looser tolerance.
    assert torch.allclose(dev_in.cpu(), cpu_in, atol=1e-3), "grad_input mismatch"
    assert torch.allclose(dev_w.cpu(), cpu_w, atol=1e-2), "grad_weight mismatch"
    assert torch.allclose(dev_b.cpu(), cpu_b, atol=1e-2), "grad_bias mismatch"


def test_linear_forward_non_contiguous_input():
    """A transposed activation makes the *forward* input non-contiguous.

    This is the layout autograd produces inside an attention block, and the one
    that made the vendor forward's ``input.view(M, K)`` raise.
    """
    layer = nn.Linear(64, 64).to(DEVICE)
    x = torch.randn(2, 64, 16, device=DEVICE, requires_grad=True)
    out = layer(x.transpose(1, 2))
    out.transpose(1, 2).sum().backward()
    torch_fl.flagos.synchronize()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_linear_backward_non_contiguous_grad_output():
    """A transposed *output* makes grad_output non-contiguous.

    Distinct from the forward case above: here the input is contiguous and it
    is the incoming gradient that is strided, which is what reaches
    linear_backward's own flattening. A plain ``.sum().backward()`` produces a
    contiguous gradient and does not exercise this.
    """
    layer = nn.Linear(64, 128).to(DEVICE)
    x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
    (layer(x).transpose(0, 1) * 2).sum().backward()
    torch_fl.flagos.synchronize()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()

    # Same computation on CPU, to check the contiguous copy did not reorder.
    torch.manual_seed(0)
    cpu_layer = nn.Linear(64, 128)
    cpu_layer.load_state_dict({k: v.cpu() for k, v in layer.state_dict().items()})
    xc = x.detach().cpu().requires_grad_(True)
    (cpu_layer(xc).transpose(0, 1) * 2).sum().backward()
    assert torch.allclose(x.grad.cpu(), xc.grad, atol=1e-3)


def test_transformer_block_trains():
    """A minimal attention+MLP step, i.e. the case that regressed.

    Guards the whole path rather than the single op: every Linear here is 3-D,
    so a flattening bug in linear_backward takes the step down.
    """
    torch.manual_seed(0)
    dim = 64
    qkv = nn.Linear(dim, 3 * dim).to(DEVICE)
    proj = nn.Linear(dim, dim).to(DEVICE)
    norm = nn.LayerNorm(dim).to(DEVICE)
    opt = torch.optim.AdamW(
        list(qkv.parameters()) + list(proj.parameters()) + list(norm.parameters()),
        lr=1e-3,
    )

    x = torch.randn(2, 16, dim, device=DEVICE)
    q, k, v = qkv(norm(x)).chunk(3, dim=-1)
    attn = torch.softmax(q @ k.transpose(-2, -1) / dim**0.5, dim=-1)
    loss = proj(attn @ v).sum()
    loss.backward()
    opt.step()
    torch_fl.flagos.synchronize()

    assert torch.isfinite(loss), "loss went non-finite"
    assert torch.isfinite(qkv.weight.grad).all(), "qkv grad non-finite"
    assert torch.isfinite(proj.weight.grad).all(), "proj grad non-finite"
