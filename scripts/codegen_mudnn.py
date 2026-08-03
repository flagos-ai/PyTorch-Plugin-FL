#!/usr/bin/env python3
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
Codegen for torch_fl Moore Threads MUSA (mudnn) operators.

Same problem as Ascend and GCU: there is no vendor dispatch key to box into (the
MUSA toolkit ships no CUDA runtime), so every kernel calls the vendor op library
-- libmudnn.so -- directly. The call shape is uniform per *category*, so this
generator is category-driven, mirroring scripts/codegen_gcu.py.

This replaces the earlier codegen_musa.py, which emitted 1186 passthroughs to
torch_musa's flat `at::musa::*` API. That API lives in libmusa_python.so, which
links against torch and therefore embeds torch's C++ object layout -- pinning
the plugin to one exact torch build (sizeof(c10::MessageLogger) went 408 -> 400
between 2.9.1 and 2.10, which corrupts the vendor .so's stack). mudnn pulls in
no torch symbols at all, so this route is torch-version-agnostic.

Unlike topsaten, mudnn is configured then run rather than called in one shot:

    Unary op;
    op.SetMode(Unary::Mode::ADD);
    op.SetAlpha(alpha);
    op.Run(handle, out, self);

Two mudnn properties shape the templates, both verified directly against the
library rather than assumed:

  - mudnn Tensors carry strides on *both* operands, and honour 0-strides. So
    broadcasting is expressed with expand() alone (a view) and non-contiguous
    inputs are read in place -- no `.contiguous()` materialization anywhere,
    which is where the GCU templates have to spend an extra copy.
  - int64 works across Unary/Binary/Reduce/MatMul. topsaten has no int64
    kernels at all, so the GCU templates carry an int64 CPU-fallback branch;
    here only genuinely unmapped dtypes (complex, quantized) fall back.

Generates:
  - csrc/aten/backends/musa/generated/musa_kernels.cc
      the kernels + REGISTER_IMPL_TO_DISPATCHER(..., Backend::kMusa, ...)
  - csrc/aten/backends/musa/generated/musa_register.inc
      the m.impl() subset for register.cc. MUSA registers PrivateUse1 ONLY for
      ops it has a kernel for; everything else stays unregistered and reaches
      the cpu_fallback (registering an op with no kernel behind it would instead
      hit the dispatcher's "backend not registered" check).
  - appends `<op> = musa` to torch_fl/configs/backends_musa.conf
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Reuse the authoritative symbol-naming from the CUDA codegen so the emitted
# REGISTER_IMPL_TO_DISPATCHER(FnType, dispatcher, ...) matches the
# DECLARE_DISPATCHER in generated/ops.h exactly (else the build won't link).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen_ops import schema_to_cpp_name  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_CC = REPO / "csrc/aten/backends/musa/generated/musa_kernels.cc"
OUT_INC = REPO / "csrc/aten/backends/musa/generated/musa_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"
CONF = REPO / "torch_fl/configs/backends_musa.conf"

# --------------------------------------------------------------------------
# Op registry: schema op name -> (category, mudnn mode).
#
# The mode is the enum entry inside the category's op class, e.g. "ABS" means
# Unary::Mode::ABS. Operand order for the alpha-carrying modes was confirmed on
# device: SUB/DIV/POW/FLOORMOD are `self OP alpha`, and *_BY_ALPHA is the
# reverse -- so no scalar overload silently computes backwards.
# --------------------------------------------------------------------------
OPS = {
    # ---- unary: Unary::Mode::<M>, Run(out, self) ----
    "abs": ("unary", "ABS"),
    "sqrt": ("unary", "SQRT"),
    "rsqrt": ("unary", "RSQRT"),
    "exp": ("unary", "EXP"),
    "log": ("unary", "LOG"),
    "log2": ("unary", "LOG2"),
    "log10": ("unary", "LOG10"),
    "log1p": ("unary", "LOG1P"),
    "sin": ("unary", "SIN"),
    "cos": ("unary", "COS"),
    "acos": ("unary", "ACOS"),
    "atan": ("unary", "ATAN"),
    "tanh": ("unary", "TANH"),
    "sigmoid": ("unary", "SIGMOID"),
    "silu": ("unary", "SILU"),
    "relu": ("unary", "RELU"),
    "reciprocal": ("unary", "RECIPROCAL"),
    "erf": ("unary", "ERF"),
    "floor": ("unary", "FLOOR"),
    "ceil": ("unary", "CEIL"),
    "sign": ("unary", "SIGN"),
    # ---- unary composed from a mode + a fixed alpha (see notes below) ----
    # mudnn has no NEG or TRUNC mode. Both fall out of an alpha op exactly:
    # neg == self * -1, trunc == truncating-divide by 1. Verified against CPU
    # on float and int64 (-3 -1 0 1 2 5 -> 3 1 0 -1 -2 -5; -2.7 -> -2).
    "neg": ("unary_alpha_const", ("MUL", "-1")),
    "trunc": ("unary_alpha_const", ("TRUNCATEDIV", "1")),
    # expm1 == exp(self) - 1, two passes over the output. Matches CPU expm1 to
    # printed precision across [-2.7, 2.7].
    "expm1": ("unary_two_pass", ("EXP", "SUB", "1")),
    # ---- binary: Binary::Mode::<M>, Run(out, self, other) ----
    "mul.Tensor": ("binary", "MUL"),
    "div.Tensor": ("binary", "TRUEDIV"),
    "maximum": ("binary", "MAX"),
    "minimum": ("binary", "MIN"),
    "remainder.Tensor": ("binary", "FLOORMOD"),
    "fmod.Tensor": ("binary", "TRUNCATEMOD"),
    "pow.Tensor_Tensor": ("binary", "POW"),
    # ---- binary_alpha: ADD_ALPHA/SUB_ALPHA carry aten's `alpha` ----
    "add.Tensor": ("binary_alpha", "ADD_ALPHA"),
    "sub.Tensor": ("binary_alpha", "SUB_ALPHA"),
    # ---- binary_cmp: bool out ----
    "eq.Tensor": ("binary_cmp", "EQ"),
    "ne.Tensor": ("binary_cmp", "NE"),
    "lt.Tensor": ("binary_cmp", "LT"),
    "gt.Tensor": ("binary_cmp", "GT"),
    "le.Tensor": ("binary_cmp", "LE"),
    "ge.Tensor": ("binary_cmp", "GE"),
    "logical_and": ("binary_cmp", "LOGICAL_AND"),
    "logical_or": ("binary_cmp", "LOGICAL_OR"),
    # ---- scalar overloads: Unary + SetAlpha, no device tensor needed ----
    # GCU has to stage these scalars into a full-size device tensor because the
    # topsaten scalar path fails in its driver; mudnn takes the scalar directly.
    "mul.Scalar": ("unary_scalar", "MUL"),
    "div.Scalar": ("unary_scalar", "TRUEDIV"),
    "pow.Tensor_Scalar": ("unary_scalar", "POW"),
    "remainder.Scalar": ("unary_scalar", "FLOORMOD"),
    "fmod.Scalar": ("unary_scalar", "TRUNCATEMOD"),
    # add.Scalar/sub.Scalar also take aten's `alpha`, which folds into the
    # scalar (self + other*alpha), so one Unary call still suffices.
    "add.Scalar": ("unary_scalar_alpha", "ADD"),
    "sub.Scalar": ("unary_scalar_alpha", "SUB"),
    # ---- scalar comparisons -> bool out ----
    "eq.Scalar": ("unary_scalar_cmp", "EQ"),
    "ne.Scalar": ("unary_scalar_cmp", "NE"),
    "lt.Scalar": ("unary_scalar_cmp", "LT"),
    "gt.Scalar": ("unary_scalar_cmp", "GT"),
    "le.Scalar": ("unary_scalar_cmp", "LE"),
    "ge.Scalar": ("unary_scalar_cmp", "GE"),
    # ---- matmul ----
    "mm": ("matmul", "MatMul"),
    "bmm": ("matmul", "BatchMatMul"),
    "mm.out": ("matmul_out", "MatMul"),
    "bmm.out": ("matmul_out", "BatchMatMul"),
    # ---- reduce over dims with optional out dtype ----
    "sum.dim_IntList": ("reduce_dims_dtype", "ADD"),
    "mean.dim": ("reduce_dims_dtype", "MEAN"),
    # ---- reduce whole tensor ----
    "sum": ("reduce_all_dtype", "ADD"),
    "mean": ("reduce_all_dtype", "MEAN"),
    # ---- misc ----
    "gelu": ("gelu", "GELU"),
    "_softmax": ("softmax_fwd", "SOFTMAX"),
    # ---- P1: further unary modes (all verified against CPU) ----
    "tan": ("unary", "TAN"),
    # mudnn's ROUND is half-to-even, which is what aten does too: -2.5 -> -2,
    # 0.5 -> 0, 3.5 -> 4. No correction needed.
    "round": ("unary", "ROUND"),
    "mish": ("unary", "MISH"),
    "hardswish": ("unary", "HARDSWISH"),
    # IS_NAN/IS_INF write a BOOL output, so they use the permissive predicate.
    "isnan": ("unary_cmp", "IS_NAN"),
    "isinf": ("unary_cmp", "IS_INF"),
    "hardsigmoid": ("unary_two_const", ("HARDSIGMOID", "1.0 / 6.0", "0.5")),
    "leaky_relu": ("unary_param", ("LEAKY_RELU", "negative_slope")),
    "elu": ("elu", "ELU"),
    "softplus": ("softplus", "SOFTPLUS"),
    "clamp": ("clamp", "CLIP"),
    "clamp_min": (
        "clamp_one_sided",
        ("CLIP", "min", "min.to<double>()", "std::numeric_limits<double>::infinity()"),
    ),
    "clamp_max": (
        "clamp_one_sided",
        ("CLIP", "max", "-std::numeric_limits<double>::infinity()", "max.to<double>()"),
    ),
    "logical_xor": ("binary_cmp", "LOGICAL_XOR"),
    "floor_divide": ("binary", "FLOORDIV"),
    # ---- P1: activation backwards ----
    # SIGMOID_BW/TANH_BW take (grad, output); aten passes `output` in that slot
    # too, so the template is shared. The rest take (grad, input).
    "sigmoid_backward": ("binary_bw", "SIGMOID_BW"),
    "tanh_backward": ("binary_bw", "TANH_BW"),
    "silu_backward": ("binary_bw", "SILU_BW"),
    "gelu_backward": ("gelu_bw", "GELU_NONE_BW"),
    "threshold_backward": ("threshold_bw", "THRESHOLD_BW"),
    "leaky_relu_backward": ("leaky_relu_bw", "LEAKY_RELU_BW"),
    # ---- P1: ternary ----
    "addcmul": ("ternary_value", "ADDCMUL_ALPHA"),
    "addcdiv": ("ternary_value", "ADDCDIV_ALPHA"),
    "where.self": ("where", "SELECT"),
    # ---- P1: addmm family (three-branch, see T_ADDMM) ----
    "addmm": ("addmm", "MatMul"),
    "baddbmm": ("addmm", "BatchMatMul"),
}

# Ops in the GCU coverage set with no mudnn equivalent. Deliberately absent from
# OPS rather than registered-and-broken: an unregistered op reaches the
# cpu_fallback and stays correct, so these keep working, just on the host.
#
# mudnn's Unary::Mode has SIN/COS/TAN/ACOS/ATAN but no SINH/COSH/ASIN, and
# composing them from EXP (sinh = (e^x - e^-x)/2) would need several passes with
# worse accuracy than the host, so it is not worth it for these three.
# relu6 is CompositeImplicitAutograd -- it has no wrapper in register.inc and
# decomposes into hardtanh, so mudnn's RELU6 mode is unreachable from here.
NO_MUDNN_EQUIVALENT = ["sinh", "cosh", "asin"]

# Ops handwritten elsewhere for MUSA would double-register the kMusa slot (which
# crashes at import), so they must be excluded here. MudnnCopy is not listed: it
# is called directly by copy_ops.cc/contiguous_ops.cc, not registered as a
# kernel.
SKIP: set = set()

# Handwritten kernels that still need an m.impl() line emitted into
# musa_register.inc. The two `*_overrideable` convolution ops cannot be left to
# the cpu_fallback like other uncovered ops -- ATen's default for them is a
# raising TORCH_CHECK, not something boxable -- so mudnn_conv.cc implements them
# and they are registered here. Kernel bodies live in
# csrc/aten/backends/musa/mudnn_conv.cc, so they are absent from OPS.
HANDWRITTEN_REGISTRATIONS = [
    "convolution_overrideable",
    "convolution_backward_overrideable",
]

# The CPU-fallback path in each kernel calls back into at::<name>. That is the
# op base name except where the base is not a real at:: function.
AT_OP_OVERRIDES = {
    "mm.out": "mm",
    "bmm.out": "bmm",
    "_softmax": "_softmax",
}

# ==========================================================================
# Templates
#
# `musa_ops` is the helper namespace from ../mudnn_common.h. EXEC_MUDNN_CMD
# takes the whole Run() expression (not op + args) because mudnn's entry points
# vary: Run, Run-with-workspace, RunWithIndices. `_mudnn_h` is the cached
# per-device Handle the macro binds.
# ==========================================================================

T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# A unary aten op that mudnn only expresses as a mode plus a constant alpha
# (neg -> MUL by -1, trunc -> TRUNCATEDIV by 1). SetAlpha is overloaded on
# double vs int64_t and the op reads the member back as the tensor's dtype, so
# an integral tensor must set the integral overload.
T_UNARY_ALPHA_CONST = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  if (at::isIntegralType(self.scalar_type(), true)) {{
    op.SetAlpha(static_cast<int64_t>({alpha}));
  }} else {{
    op.SetAlpha(static_cast<double>({alpha}));
  }}
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Two mudnn passes for one aten op (expm1 = EXP then SUB alpha=1). The second
# pass reads and writes `out`, which mudnn allows for elementwise unary.
T_UNARY_TWO_PASS = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary first;
  first.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self, first.Run(_mudnn_h, t_out.get(), t_self.get()));
  musa_ops::MudnnTensorWrapper t_out2(out);
  musa_ops::MudnnTensorWrapper t_in2(out);
  musa_ops::mudnn::Unary second;
  second.SetMode(musa_ops::mudnn::Unary::Mode::{mode2});
  if (at::isIntegralType(self.scalar_type(), true)) {{
    second.SetAlpha(static_cast<int64_t>({alpha}));
  }} else {{
    second.SetAlpha(static_cast<double>({alpha}));
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self, second.Run(_mudnn_h, t_out2.get(), t_in2.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Output dtype follows at::result_type, matching PyTorch's promotion.
#
# `other` may be a CPU tensor: PyTorch wraps a Python number operand into a
# 0-dim CPU tensor and dispatches through the Tensor overload (a * 3.0 ->
# mul.Tensor). Handing that host pointer to mudnn would fault, so any non-device
# operand is moved onto self's device first.
#
# expand() to the broadcast shape stays a view (it only introduces 0-strides),
# and mudnn reads 0-strides correctly -- verified: (2,3) + (3,) via strides
# {0,1} gives 11 22 33 14 25 36. So unlike the GCU templates there is no
# `.contiguous()` here and broadcasting costs no allocation.
_BINARY_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto other_c = other.to(self.device(), result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  auto self_b = self_c.expand(out_shape);
  auto other_b = other_c.expand(out_shape);
"""

T_BINARY = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

T_BINARY_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu(), alpha).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(alpha.to<int64_t>());
  }} else {{
    op.SetAlpha(alpha.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

T_BINARY_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(at::kBool));

  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# A Scalar participates in promotion only via its category (integral scalars do
# not widen a float tensor), which is exactly at::result_type(Tensor, Scalar).
#
# The scalar goes straight into Unary::SetAlpha -- one kernel, no staging
# tensor. Operand order is `self OP alpha` for every mode used here (SUB, DIV,
# POW, FLOORMOD, TRUNCATEMOD all confirmed on device; the reverse spellings are
# the separate *_BY_ALPHA modes, which we do not use).
_SCALAR_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
"""

# Braces stay doubled: this fragment is concatenated into a template that is
# .format()ed once, at generation time.
_SCALAR_SET_ALPHA = """\
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(other.to<int64_t>());
  }} else {{
    op.SetAlpha(other.to<double>());
  }}
"""

T_UNARY_SCALAR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
"""
    + _SCALAR_SET_ALPHA
    + """\
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# add.Scalar/sub.Scalar: aten's alpha scales the scalar operand, so
# self + other*alpha folds into a single ADD/SUB alpha. Folded on the host in
# the scalar's own type to avoid a needless int->double round trip.
T_UNARY_SCALAR_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other, alpha).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(other.to<int64_t>() * alpha.to<int64_t>());
  }} else {{
    op.SetAlpha(other.to<double>() * alpha.to<double>());
  }}
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# Tensor-vs-Scalar comparison. The comparison itself happens in the promoted
# type, but the result is always bool -- confirmed on device (GT alpha=3 over
# 1..6 gives 0 0 0 1 1 1 into a BOOL tensor).
T_UNARY_SCALAR_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
"""
    + _SCALAR_SET_ALPHA
    + """\
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# mm/bmm. mudnn's MatMul and BatchMatMul both want a workspace maintainer (the
# query returned 0 bytes for the shapes probed, but the argument is mandatory).
# Operands are made contiguous here: unlike the elementwise ops, a GEMM's inner
# layout requirements are not something the stride descriptor alone guarantees,
# so this keeps mm(a.t(), b) correct rather than fast.
T_MATMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = at::empty(out_shape, self.options());
  auto self_c = self.contiguous();
  auto mat2_c = mat2.contiguous();

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_mat2(mat2_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::{mode} op;
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_mat2.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

T_MATMUL_OUT = """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat2.scalar_type())) {{
    out.copy_(at::{at_op}(self.cpu(), mat2.cpu()));
    return out;
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}
  auto self_c = self.contiguous();
  auto mat2_c = mat2.contiguous();

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_mat2(mat2_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::{mode} op;
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_mat2.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# sum/mean over an optional dim list. An absent or empty list reduces every
# dim; negative dims are wrapped. Dims are erased high-to-low so an earlier
# erase does not shift a later index. sum promotes integral inputs to int64,
# matching PyTorch -- and unlike topsaten, mudnn *can* reduce in int64
# (verified: ADD over int64 {1..6} by dim gives 6, 15), so integral sums stay
# on device instead of taking a CPU fallback.
T_REDUCE_DIMS_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      (dtype.has_value() && !musa_ops::{dtype_pred}(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  if (dim.has_value() && !dim.value().empty()) {{
    for (int64_t d : dim.value()) norm_dims.push_back(d < 0 ? d + ndim : d);
  }} else {{
    for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  }}
  // aten's answer shape, and the *squeezed* shape mudnn must be handed. mudnn
  // silently drops all but the first output element when the output still
  // carries the reduced axes as extent-1 dims and the input is non-contiguous
  // (measured on v3300: (4,5) stride (0,1) reduced over dim 0 into a [1,5]
  // output writes only out[0]). Reducing into the squeezed shape is correct in
  // every configuration probed, so keepdim is restored with a view afterwards.
  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(norm_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  auto squeezed_shape = self.sizes().vec();
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
    squeezed_shape.erase(squeezed_shape.begin() + d);
  }}

  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  // mudnn multi-dim Reduce (more than one axis at a time) silently ignores
  // strides and reads the input as if contiguous (verified: (4,5) stride (0,1)
  // reduced over all dims gives 210 = sum(1..20) instead of 60 = 4*sum(1..5)).
  // Single-dim reduces honour strides correctly. Materializing once fixes both
  // the stride bug and the SIGFPE on fully-broadcast multi-dim reduces.
  if (norm_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceWouldFault(self_c, norm_dims.size())) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty(squeezed_shape, self.options().dtype(out_dtype));
  std::vector<int> mudnn_dims = musa_ops::ToMudnnDims(norm_dims, ndim);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

T_REDUCE_ALL_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      (dtype.has_value() && !musa_ops::{dtype_pred}(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dtype).to(self.device());
  }}
  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  std::vector<int> mudnn_dims;
  for (int64_t d = 0; d < self.dim(); ++d) {{
    mudnn_dims.push_back(static_cast<int>(d));
  }}
  // Reducing every dim at once is a multi-dim Reduce, which mudnn runs as if
  // the input were contiguous -- it ignores strides outright, and faults on a
  // fully-broadcast input. Materializing once covers both. Same measurement as
  // in T_REDUCE_DIMS_DTYPE.
  if (mudnn_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceWouldFault(self_c, mudnn_dims.size())) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty({{}}, self.options().dtype(out_dtype));

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# aten spells the variant as a string; mudnn has separate modes for it.
T_GELU = """\
at::Tensor {kernel}(const at::Tensor& self, c10::string_view approximate) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), approximate).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());

  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(
      approximate == "tanh" ? musa_ops::mudnn::Unary::Mode::GELU_TANH
                            : musa_ops::mudnn::Unary::Mode::GELU);
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# `half_to_float` asks for a float output from a half input, which mudnn's
# Softmax does not express (out dtype must match in), so that combination runs
# on the host. ACCURATE is the max-subtracting algorithm, matching aten.
T_SOFTMAX_FWD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) || half_to_float) {{
    return at::{at_op}(self.cpu(), dim, half_to_float).to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = at::empty(self.sizes(), self.options());

  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Softmax op;
  op.SetMode(musa_ops::mudnn::Softmax::Mode::{mode});
  op.SetAlgorithm(musa_ops::mudnn::Softmax::Algorithm::ACCURATE);
  op.SetDim(static_cast<int>(d));
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Activation backwards. mudnn spells these as Binary modes taking the gradient
# and one saved tensor, so the whole family is one template plus a mode name.
#
# Which saved tensor is the second operand differs per mode and is NOT visible
# in the header -- both orders return SUCCESS, only the numbers differ. Measured
# against CPU formulas: SIGMOID_BW/TANH_BW consume the op's *output*
# (g*y*(1-y), g*(1-y^2)), every other mode here consumes the *input*. aten
# passes exactly that tensor in the same position either way, so `self` maps
# straight through; the distinction is only documented, never branched on.
T_BINARY_BW = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu()).to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# gelu_backward: aten spells the variant as a string, mudnn as two modes.
T_GELU_BW = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu(), approximate)
        .to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(
      approximate == "tanh" ? musa_ops::mudnn::Binary::Mode::GELU_TANH_BW
                            : musa_ops::mudnn::Binary::Mode::GELU_NONE_BW);
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# threshold_backward(grad_output, self, threshold). mudnn's THRESHOLD_BW takes
# the threshold as alpha; measured to match aten (grad passes where self >
# threshold, 0 elsewhere).
T_THRESHOLD_BW = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu(), threshold)
        .to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(threshold.to<int64_t>());
  }} else {{
    op.SetAlpha(threshold.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# leaky_relu_backward(grad_output, self, negative_slope, self_is_result).
# LEAKY_RELU_BW's alpha is the slope. `self_is_result` only tells autograd
# whether `self` is the output rather than the input; for a leaky ReLU the
# gradient test (`> 0`) gives the same answer either way, so it is unused.
T_LEAKY_RELU_BW = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(
               grad_output.cpu(), self.cpu(), negative_slope, self_is_result)
        .to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(negative_slope.to<int64_t>());
  }} else {{
    op.SetAlpha(negative_slope.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Ternary ops. Measured to map onto aten 1:1 with Run(out, self, t1, t2):
# ADDCMUL_ALPHA is self + value*t1*t2, ADDCDIV_ALPHA is self + value*t1/t2.
# All three operands broadcast against each other, and mudnn honours the
# resulting 0-strides, so expand() alone is enough.
T_TERNARY_VALUE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(tensor1.scalar_type()) ||
      !musa_ops::{dtype_pred}(tensor2.scalar_type())) {{
    return at::{at_op}(self.cpu(), tensor1.cpu(), tensor2.cpu(), value)
        .to(self.device());
  }}
  auto result_dtype = at::result_type(self, tensor1);
  result_dtype = at::promoteTypes(result_dtype, tensor2.scalar_type());
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto t1_c = tensor1.to(self.device(), result_dtype);
  auto t2_c = tensor2.to(self.device(), result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), t1_c.sizes());
  out_shape = at::infer_size(out_shape, t2_c.sizes());
  auto self_b = self_c.expand(out_shape);
  auto t1_b = t1_c.expand(out_shape);
  auto t2_b = t2_c.expand(out_shape);
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_t1(t1_b);
  musa_ops::MudnnTensorWrapper t_t2(t2_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Ternary op;
  op.SetMode(musa_ops::mudnn::Ternary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(value.to<int64_t>());
  }} else {{
    op.SetAlpha(value.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_t1.get(), t_t2.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# where.self(condition, self, other) -> Ternary::SELECT(mask, x, y). The
# condition is bool and stays bool; only the value operands promote.
T_WHERE = """\
at::Tensor {kernel}(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type()) ||
      condition.scalar_type() != at::kBool) {{
    return at::{at_op}(condition.cpu(), self.cpu(), other.cpu())
        .to(self.device());
  }}
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.to(condition.device(), result_dtype);
  auto other_c = other.to(condition.device(), result_dtype);
  auto out_shape = at::infer_size(condition.sizes(), self_c.sizes());
  out_shape = at::infer_size(out_shape, other_c.sizes());
  auto cond_b = condition.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto other_b = other_c.expand(out_shape);
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));

  musa_ops::MudnnTensorWrapper t_cond(cond_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Ternary op;
  op.SetMode(musa_ops::mudnn::Ternary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_cond.get(), t_self.get(),
             t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# addmm/baddbmm: out = beta*self + alpha*(mat1 @ mat2).
#
# mudnn computes d = alpha*a@b + beta*c + gamma*bias, and which slot `self`
# takes depends on its shape -- this mirrors torch_musa's own three-branch
# dispatch (csrc/aten/ops/Matmul.cpp), verified numerically here:
#   * self shaped like out -> the C slot, aten's beta -> SetBeta
#   * self is 1-D of length N -> the bias slot with c aliasing d, so aten's
#     beta must ride on *gamma* instead (SetBeta stays 0, or it would fold in
#     the output buffer's prior contents)
#   * anything else (a scalar, [M,1]) -> plain Run, then add the bias on the
#     host side with a normal aten add.
#
# MatMul rejects non-contiguous operands ("MatMulRun only support contiguous
# tensor"), unlike the elementwise ops, so mat1/mat2/self are materialized.
T_ADDMM = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat1.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
        .to(self.device());
  }}
  std::vector<int64_t> out_shape = mat1.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = at::empty(out_shape, mat1.options());
  auto mat1_c = mat1.contiguous();
  auto mat2_c = mat2.contiguous();

  musa_ops::MudnnTensorWrapper t_mat1(mat1_c);
  musa_ops::MudnnTensorWrapper t_mat2(mat2_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::{mode} op;

  const bool self_is_out_shaped = self.sizes().equals(out_shape);
  const bool self_is_vector =
      self.dim() == 1 && self.size(0) == out_shape.back();

  if (self_is_out_shaped) {{
    auto self_c = self.to(mat1.device(), mat1.scalar_type()).contiguous();
    musa_ops::MudnnTensorWrapper t_self(self_c);
    musa_ops::mudnn::Tensor empty_bias;
    op.SetAlpha(alpha.to<double>());
    op.SetBeta(beta.to<double>());
    op.SetGamma(1.0);
    EXEC_MUDNN_CMD(
        "{at_op}", self,
        op.RunWithBiasAdd(_mudnn_h, t_out.get(), t_mat1.get(), t_mat2.get(),
                          t_self.get(), empty_bias,
                          musa_ops::MudnnWorkspaceFor(out)));
  }} else if (self_is_vector) {{
    auto self_c = self.to(mat1.device(), mat1.scalar_type()).contiguous();
    musa_ops::MudnnTensorWrapper t_self(self_c);
    // c aliases d here, so beta must stay 0 and aten's beta rides on gamma.
    op.SetAlpha(alpha.to<double>());
    op.SetBeta(0.0);
    op.SetGamma(beta.to<double>());
    EXEC_MUDNN_CMD(
        "{at_op}", self,
        op.RunWithBiasAdd(_mudnn_h, t_out.get(), t_mat1.get(), t_mat2.get(),
                          t_out.get(), t_self.get(),
                          musa_ops::MudnnWorkspaceFor(out)));
  }} else {{
    op.SetAlpha(alpha.to<double>());
    op.SetBeta(0.0);
    op.SetGamma(1.0);
    EXEC_MUDNN_CMD(
        "{at_op}", self,
        op.Run(_mudnn_h, t_out.get(), t_mat1.get(), t_mat2.get(),
               musa_ops::MudnnWorkspaceFor(out)));
    out.add_(self, beta);
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# A unary predicate: same call shape as T_UNARY but the output is bool
# (isnan, isinf). Verified that mudnn accepts a BOOL destination for these.
T_UNARY_CMP = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# A unary mode configured by two fixed constants rather than by aten arguments.
# mudnn's HARDSIGMOID is clamp(alpha*x + beta, 0, 1) with both defaulting to 0,
# so leaving them unset returns all zeros. aten's hardsigmoid is alpha=1/6,
# beta=0.5 (verified against CPU).
T_UNARY_TWO_CONST = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(static_cast<double>({alpha}));
  op.SetBeta(static_cast<double>({beta}));
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# leaky_relu(self, negative_slope): one aten Scalar straight into alpha.
T_UNARY_PARAM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& {param}) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), {param}).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha({param}.to<double>());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# clamp_min / clamp_max map onto CLIP, whose alpha is the lower bound and beta
# the upper. Both default to 0 in mudnn, so the unused side must be set to an
# explicit infinity or the op would clip against 0.
T_CLAMP_ONE_SIDED = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& {param}) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), {param}).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha({lo});
  op.SetBeta({hi});
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# clamp(self, min?, max?) -- both bounds optional; an absent bound becomes the
# corresponding infinity.
T_CLAMP = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), min, max).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(
      min.has_value() ? min.value().to<double>()
                      : -std::numeric_limits<double>::infinity());
  op.SetBeta(
      max.has_value() ? max.value().to<double>()
                      : std::numeric_limits<double>::infinity());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# softplus(self, beta, threshold). mudnn's setter names are inverted relative
# to aten: SetAlpha carries aten's `beta`, SetBeta carries aten's `threshold`.
# Leaving either unset returns inf.
T_SOFTPLUS = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), beta, threshold).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(beta.to<double>());
  op.SetBeta(threshold.to<double>());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# elu(self, alpha, scale, input_scale) computes
# scale * (max(0,x) + min(0, alpha*(exp(input_scale*x)-1))). mudnn's ELU takes
# only alpha, so a non-unit scale or input_scale has no expression here and
# takes the CPU fallback rather than silently ignoring the argument.
T_ELU = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      scale.to<double>() != 1.0 || input_scale.to<double>() != 1.0) {{
    return at::{at_op}(self.cpu(), alpha, scale, input_scale)
        .to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(alpha.to<double>());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

CATEGORIES = {
    "unary": T_UNARY,
    "unary_alpha_const": T_UNARY_ALPHA_CONST,
    "unary_two_pass": T_UNARY_TWO_PASS,
    "binary": T_BINARY,
    "binary_alpha": T_BINARY_ALPHA,
    "binary_cmp": T_BINARY_CMP,
    "unary_scalar": T_UNARY_SCALAR,
    "unary_scalar_alpha": T_UNARY_SCALAR_ALPHA,
    "unary_scalar_cmp": T_UNARY_SCALAR_CMP,
    "matmul": T_MATMUL,
    "matmul_out": T_MATMUL_OUT,
    "reduce_dims_dtype": T_REDUCE_DIMS_DTYPE,
    "reduce_all_dtype": T_REDUCE_ALL_DTYPE,
    "gelu": T_GELU,
    "softmax_fwd": T_SOFTMAX_FWD,
    "unary_cmp": T_UNARY_CMP,
    "unary_two_const": T_UNARY_TWO_CONST,
    "unary_param": T_UNARY_PARAM,
    "clamp_one_sided": T_CLAMP_ONE_SIDED,
    "clamp": T_CLAMP,
    "softplus": T_SOFTPLUS,
    "elu": T_ELU,
    "binary_bw": T_BINARY_BW,
    "gelu_bw": T_GELU_BW,
    "threshold_bw": T_THRESHOLD_BW,
    "leaky_relu_bw": T_LEAKY_RELU_BW,
    "ternary_value": T_TERNARY_VALUE,
    "where": T_WHERE,
    "addmm": T_ADDMM,
}

# Categories whose mudnn mode is arithmetic, and therefore rejects bool operands
# ("Unsupported binary mode: MUL, with left data type: BOOL"). PyTorch does
# define bool arithmetic, so these kernels use the stricter predicate and let
# bool take the CPU fallback. The comparison and logical categories accept bool
# natively, and the reductions cast to the accumulate dtype before running.
ARITHMETIC_CATEGORIES = {
    "unary",
    "unary_alpha_const",
    "unary_two_pass",
    "binary",
    "binary_alpha",
    "unary_scalar",
    "unary_scalar_alpha",
    "matmul",
    "matmul_out",
    "gelu",
    "softmax_fwd",
    "unary_two_const",
    "unary_param",
    "clamp_one_sided",
    "clamp",
    "softplus",
    "elu",
    "binary_bw",
    "gelu_bw",
    "threshold_bw",
    "leaky_relu_bw",
    "ternary_value",
    "addmm",
}

# The mudnn class each category configures, for symbol validation.
CATEGORY_CLASS = {
    "unary": "Unary",
    "unary_alpha_const": "Unary",
    "unary_two_pass": "Unary",
    "binary": "Binary",
    "binary_alpha": "Binary",
    "binary_cmp": "Binary",
    "unary_scalar": "Unary",
    "unary_scalar_alpha": "Unary",
    "unary_scalar_cmp": "Unary",
    "matmul": None,  # mode names the class itself (MatMul / BatchMatMul)
    "matmul_out": None,
    "reduce_dims_dtype": "Reduce",
    "reduce_all_dtype": "Reduce",
    "gelu": "Unary",
    "softmax_fwd": "Softmax",
    "unary_cmp": "Unary",
    "unary_two_const": "Unary",
    "unary_param": "Unary",
    "clamp_one_sided": "Unary",
    "clamp": "Unary",
    "softplus": "Unary",
    "elu": "Unary",
    "binary_bw": "Binary",
    "gelu_bw": "Binary",
    "threshold_bw": "Binary",
    "leaky_relu_bw": "Binary",
    "ternary_value": "Ternary",
    "where": "Ternary",
    "addmm": None,  # mode names the class itself (MatMul / BatchMatMul)
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_mudnn.py -- DO NOT EDIT.
//
// mudnn kernels for the Moore Threads MUSA backend, generated per-category.
// Each kernel describes its aten tensors as musa::dnn::Tensors, configures a
// mudnn op object and issues one Run via EXEC_MUDNN_CMD. Dispatchers are
// declared in generated/ops.h (shared with the CUDA codegen); here we only fill
// the Backend::kMusa slot.
//
// mudnn links against musart only -- no torch symbols -- so unlike the previous
// at::musa::* passthroughs these kernels are independent of the torch version.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/result_type.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../mudnn_common.h"

namespace at::native::flagos {

namespace musa_ops = at::native::flagos::musa_ops;

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_mudnn.py -- DO NOT EDIT.
//
// m.impl() lines for the ops that have a mudnn kernel. Included by register.cc
// inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_MUSA is set, in place
// of the full generated/register.inc list: an op registered on PrivateUse1
// without a kernel behind it fails the dispatcher's "backend not registered"
// check, whereas an unregistered op simply reaches the cpu_fallback. So this
// file is exactly the MUSA coverage set.

"""


def libmudnn_path() -> Path:
    env = os.environ.get("MUDNN_LIB")
    if env:
        return Path(env)
    musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
    return Path(musa_home) / "lib/libmudnn.so"


def symbols(lib: Path):
    """mudnn ops are C++ classes in `namespace musa::dnn`, so the mangled names
    are demangled (nm -C) and matched as `musa::dnn::<Class>::`."""
    if not lib.exists():
        print(f"[warn] {lib} not found; skipping symbol validation", file=sys.stderr)
        return None
    out = subprocess.run(
        ["nm", "-DC", "--defined-only", str(lib)], capture_output=True, text=True
    )
    return set(re.findall(r"musa::dnn::(\w+)::", out.stdout))


def wrapper_map():
    """op name -> Wrapper<Name>, read back from the CUDA codegen's register.inc
    so the m.impl() subset we emit cannot drift from the wrappers that exist."""
    if not REGISTER_INC.exists():
        return {}
    text = REGISTER_INC.read_text()
    return dict(re.findall(r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);', text, flags=re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--category",
        default="all",
        choices=["all"] + list(CATEGORIES),
        help="restrict generation to one category (default: all)",
    )
    ap.add_argument(
        "--no-conf",
        action="store_true",
        help="do not append covered ops to backends_musa.conf",
    )
    args = ap.parse_args()

    syms = symbols(libmudnn_path())
    wrappers = wrapper_map()

    bodies = []
    covered = []  # (op, mode, category)
    skipped = []  # (op, reason)

    for op, (cat, mode) in OPS.items():
        if args.category != "all" and cat != args.category:
            continue
        if op in SKIP:
            skipped.append((op, "handwritten"))
            continue
        base = op.split(".")[0]
        # Composed categories carry a tuple of (mode, ...) rather than one mode.
        if isinstance(mode, tuple):
            mode_name, extra = mode[0], mode[1:]
        else:
            mode_name, extra = mode, ()
        cls = CATEGORY_CLASS[cat] or mode_name
        if syms is not None and cls not in syms:
            skipped.append((op, f"musa::dnn::{cls} not in {libmudnn_path().name}"))
            continue
        if op not in wrappers:
            skipped.append((op, "no wrapper in generated/register.inc"))
            continue
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelMusa"  # SqrtFn -> SqrtKernelMusa
        fmt = dict(
            kernel=kernel,
            mode=mode_name,
            fn=fn,
            disp=disp,
            at_op=AT_OP_OVERRIDES.get(op, base),
            promote_integral="true" if base == "sum" else "false",
            dtype_pred=(
                "MudnnSupportsArithmeticDtype"
                if cat in ARITHMETIC_CATEGORIES
                else "MudnnSupportsDtype"
            ),
        )
        if cat == "unary_alpha_const":
            fmt["alpha"] = extra[0]
        elif cat == "unary_two_pass":
            fmt["mode2"], fmt["alpha"] = extra[0], extra[1]
        elif cat == "unary_two_const":
            fmt["alpha"], fmt["beta"] = extra[0], extra[1]
        elif cat == "unary_param":
            fmt["param"] = extra[0]
        elif cat == "clamp_one_sided":
            fmt["param"], fmt["lo"], fmt["hi"] = extra[0], extra[1], extra[2]
        bodies.append(CATEGORIES[cat].format(**fmt))
        covered.append((op, mode_name, cat))

    for op in NO_MUDNN_EQUIVALENT:
        skipped.append((op, "no mudnn mode; stays on cpu_fallback"))

    # Handwritten kernels contribute an m.impl() line but no generated body.
    handwritten = []
    if args.category == "all":
        for op in HANDWRITTEN_REGISTRATIONS:
            if op in wrappers:
                handwritten.append(op)
            else:
                skipped.append((op, "handwritten, but no wrapper in register.inc"))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    impl_ops = sorted([op for op, _, _ in covered] + handwritten)
    impls = "".join(f'  m.impl("{op}", {wrappers[op]});\n' for op in impl_ops)
    OUT_INC.write_text(INC_HEADER + impls)

    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    print(f"[gen] {OUT_INC.relative_to(REPO)}  ({len(impl_ops)} m.impl lines)")
    for op in handwritten:
        print(f"       * {op} (handwritten in mudnn_conv.cc)")
    by_cat = {}
    for op, mode, cat in covered:
        by_cat.setdefault(cat, []).append((op, mode))
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if items:
            print(f"    [{cat}] {len(items)}")
            for op, mode in items:
                print(f"       + {op} -> {mode}")
    for op, why in skipped:
        print(f"       - {op} skipped ({why})")

    if not args.no_conf and covered:
        existing = CONF.read_text() if CONF.exists() else ""
        # Strip any prior codegen block so re-runs stay idempotent.
        marker = "\n# --- generated by codegen_mudnn.py"
        if marker in existing:
            existing = existing[: existing.index(marker)].rstrip() + "\n"
        lines = []
        for op in [o for o, _, _ in covered] + handwritten:
            if f"\n{op} = " not in ("\n" + existing):
                lines.append(f"{op} = musa")
        new = existing.rstrip() + "\n"
        if lines:
            new += "\n# --- generated by codegen_mudnn.py ---\n"
            new += "\n".join(lines) + "\n"
        CONF.write_text(new)
        print(f"[conf] wrote {len(lines)} generated op(s) to {CONF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
