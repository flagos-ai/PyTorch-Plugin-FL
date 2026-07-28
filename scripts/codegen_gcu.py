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
Codegen for torch_fl Enflame GCU (topsaten) operators.

Same problem as Ascend: there is no vendor dispatch key to box into (no CUDA
runtime exists on GCU at all), so every kernel must call the vendor op library
-- libtopsaten.so -- directly. And as with Ascend the call shape is uniform per
*category*, so this generator is category-driven.

topsaten is simpler than aclnn: one direct call, no workspace/executor phase.

    topsaten::topsatenAdd(out, lhs, rhs, alpha, stream)

Generates:
  - csrc/aten/backends/gcu/generated/gcu_kernels.cc
      the kernels + REGISTER_IMPL_TO_DISPATCHER(..., Backend::kGcu, ...)
  - csrc/aten/backends/gcu/generated/gcu_register.inc
      the m.impl() subset for register.cc. GCU registers PrivateUse1 ONLY for
      ops it has a kernel for; everything else stays unregistered and reaches
      the cpu_fallback (registering all 2033 ops would instead hit the
      dispatcher's "backend not registered" check, since neither the CUDA
      boxing kernels nor FlagGems are built here).
  - appends `<op> = gcu` to torch_fl/configs/backends_gcu.conf

Validation:
  - the derived topsaten<Name> must exist in libtopsaten.so or the op is
    skipped with a warning. Symbols are C++-mangled (namespace topsaten), so
    they are read via `nm -DC` and matched as `topsaten::topsaten<Name>`.
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
OUT_CC = REPO / "csrc/aten/backends/gcu/generated/gcu_kernels.cc"
OUT_INC = REPO / "csrc/aten/backends/gcu/generated/gcu_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"
CONF = REPO / "torch_fl/configs/backends_gcu.conf"

# --------------------------------------------------------------------------
# Op registry: schema op name -> (category, topsaten-name override or None).
#
# Default topsaten name = "topsaten" + PascalCase(op base); a non-None override
# replaces that stem for irregular spellings (_softmax -> SoftmaxForward).
# --------------------------------------------------------------------------
OPS = {
    # ---- unary: topsaten<Name>(out, self) ----
    "abs": ("unary", None),
    "sqrt": ("unary", None),
    "rsqrt": ("unary", None),
    "exp": ("unary", None),
    "expm1": ("unary", None),
    "log": ("unary", None),
    "log2": ("unary", None),
    "log10": ("unary", None),
    "log1p": ("unary", None),
    "sin": ("unary", None),
    "cos": ("unary", None),
    "sinh": ("unary", None),
    "cosh": ("unary", None),
    "asin": ("unary", None),
    "acos": ("unary", None),
    "atan": ("unary", None),
    "tanh": ("unary", None),
    "sigmoid": ("unary", None),
    "silu": ("unary", None),
    "relu": ("unary", None),
    "neg": ("unary", None),
    "reciprocal": ("unary", None),
    "erf": ("unary", None),
    "floor": ("unary", None),
    "ceil": ("unary", None),
    "trunc": ("unary", None),
    "sign": ("unary", None),
    # ---- binary: topsaten<Name>(out, self, other) ----
    "mul.Tensor": ("binary", None),
    "div.Tensor": ("binary", None),
    "maximum": ("binary", None),
    "minimum": ("binary", None),
    "remainder.Tensor": ("binary", None),
    "fmod.Tensor": ("binary", None),
    "pow.Tensor_Tensor": ("binary", None),
    # ---- binary_alpha: topsaten<Name>(out, self, other, alpha) ----
    "add.Tensor": ("binary_alpha", None),
    "sub.Tensor": ("binary_alpha", None),
    # ---- binary_cmp: bool out ----
    "eq.Tensor": ("binary_cmp", None),
    "ne.Tensor": ("binary_cmp", None),
    "lt.Tensor": ("binary_cmp", None),
    "gt.Tensor": ("binary_cmp", None),
    "le.Tensor": ("binary_cmp", None),
    "ge.Tensor": ("binary_cmp", None),
    "logical_and": ("binary_cmp", None),
    "logical_or": ("binary_cmp", None),
    # ---- binary_scalar: topsaten<Name>(out, self, scalar) ----
    "pow.Tensor_Scalar": ("binary_scalar", None),
    "remainder.Scalar": ("binary_scalar", None),
    "fmod.Scalar": ("binary_scalar", None),
    # ---- scalar staged as a device tensor (see T_BINARY_SCALAR_AS_TENSOR) ----
    "mul.Scalar": ("binary_scalar_as_tensor", None),
    "div.Scalar": ("binary_scalar_as_tensor", None),
    "add.Scalar": ("binary_scalar_alpha_as_tensor", None),
    "sub.Scalar": ("binary_scalar_alpha_as_tensor", None),
    # ---- binary_scalar_cmp: Tensor + Scalar -> bool ----
    "eq.Scalar": ("binary_scalar_cmp", None),
    "ne.Scalar": ("binary_scalar_cmp", None),
    "lt.Scalar": ("binary_scalar_cmp", None),
    "gt.Scalar": ("binary_scalar_cmp", None),
    "le.Scalar": ("binary_scalar_cmp", None),
    "ge.Scalar": ("binary_scalar_cmp", None),
    # ---- matmul ----
    "mm": ("matmul", None),
    "bmm": ("matmul", None),
    "mm.out": ("matmul_out", None),
    "bmm.out": ("matmul_out", None),
    # ---- reduce over dims with optional out dtype ----
    "sum.dim_IntList": ("reduce_dims_dtype", None),
    "mean.dim": ("reduce_dims_dtype", None),
    # ---- reduce whole tensor ----
    "sum": ("reduce_all_dtype", None),
    "mean": ("reduce_all_dtype", None),
    # ---- misc ----
    "gelu": ("gelu", None),
    "_softmax": ("softmax_fwd", "SoftmaxForward"),
}

# Ops handwritten elsewhere for GCU would double-register the kGcu slot (which
# crashes at import), so they must be excluded here. None yet.
SKIP: set = set()

# The int64 CPU-fallback path in each kernel calls back into at::<name>. That is
# the op base name except where the base is not a real at:: function.
AT_OP_OVERRIDES = {
    "mm.out": "mm",
    "bmm.out": "bmm",
    "_softmax": "_softmax",
}

# ==========================================================================
# Templates
# ==========================================================================

T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# topsaten does not broadcast for us, so inputs are expanded (and made
# contiguous, since an expanded view has 0-strides) to the common shape first.
# Output dtype follows at::result_type, matching PyTorch's promotion.
#
# `other` may be a CPU tensor: PyTorch wraps a Python number operand into a
# 0-dim CPU tensor and dispatches through the Tensor overload (a * 3.0 ->
# mul.Tensor). Handing that host pointer to topsaten fails in the driver, so any
# non-device operand is moved onto self's device first.
_BINARY_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto other_c = other.to(self.device(), result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  auto self_b = self_c.expand(out_shape).contiguous();
  auto other_b = other_c.expand(out_shape).contiguous();
"""

T_BINARY = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu(), alpha).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get(), t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(at::kBool));

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# A Scalar participates in promotion only via its category (integral scalars do
# not widen a float tensor), which is exactly at::result_type(Tensor, Scalar).
_SCALAR_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto t_other = gcu::ToTopsatenScalar(other, result_dtype);
"""

T_BINARY_SCALAR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_SCALAR_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other, alpha).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other, t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# add/sub/mul/div reject topsaten's tensor-with-scalar overload in-process (the
# scalar is staged in host memory the driver will not accept), so the scalar is
# materialized as a device tensor and the tensor-with-tensor overload is used.
_SCALAR_AS_TENSOR_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = (self.scalar_type() == result_dtype ? self : self.to(result_dtype))
                    .contiguous();
  auto other_t = gcu::ScalarToDeviceTensor(
      other, self_c.sizes(), self_c.options());
  auto out = at::empty(self_c.sizes(), self.options().dtype(result_dtype));
"""

T_BINARY_SCALAR_AS_TENSOR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_AS_TENSOR_PROLOGUE
    + """\
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_other(other_t);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_SCALAR_ALPHA_AS_TENSOR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other, alpha).to(self.device());
  }}
"""
    + _SCALAR_AS_TENSOR_PROLOGUE
    + """\
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_other(other_t);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get(), t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_SCALAR_CMP = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
  auto t_other = gcu::ToTopsatenScalar(other, self.scalar_type());

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_MATMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = at::empty(out_shape, self.options());

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mat2(mat2);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_mat2.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_MATMUL_OUT = """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat2.scalar_type())) {{
    out.copy_(at::{at_op}(self.cpu(), mat2.cpu()));
    return out;
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mat2(mat2);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_mat2.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# sum/mean over an optional dim list. An absent or empty list reduces every
# dim; negative dims are wrapped. Dims are erased high-to-low so an earlier
# erase does not shift a later index. sum promotes integral inputs to int64,
# matching PyTorch.
T_REDUCE_DIMS_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  // Integral reductions accumulate in int64 (PyTorch's rule), which topsaten
  // cannot express, so they take the CPU path along with int64 inputs.
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      ({promote_integral} && at::isIntegralType(self.scalar_type(), true)) ||
      (dtype.has_value() && !gcu::TopsatenSupportsDtype(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  if (dim.has_value() && !dim.value().empty()) {{
    for (int64_t d : dim.value()) norm_dims.push_back(d < 0 ? d + ndim : d);
  }} else {{
    for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  }}
  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(norm_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
  }}

  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = at::empty(out_shape, self.options().dtype(out_dtype));
  gcu::TopsatenSizeWrapper t_dims(norm_dims);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_dims.get(),
      keepdim, gcu::ToTopsatenDataType(out_dtype));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_REDUCE_ALL_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {{
  // See T_REDUCE_DIMS_DTYPE: int64 accumulation is not available on topsaten.
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      ({promote_integral} && at::isIntegralType(self.scalar_type(), true)) ||
      (dtype.has_value() && !gcu::TopsatenSupportsDtype(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dtype).to(self.device());
  }}
  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = at::empty({{}}, self.options().dtype(out_dtype));

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(),
      gcu::ToTopsatenDataType(out_dtype));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_GELU = """\
at::Tensor {kernel}(const at::Tensor& self, c10::string_view approximate) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), approximate).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  std::string approx(approximate);

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), approx.c_str());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_SOFTMAX_FWD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, half_to_float).to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = at::empty(self.sizes(), self.options().dtype(out_dtype));

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), d, half_to_float);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

CATEGORIES = {
    "unary": T_UNARY,
    "binary": T_BINARY,
    "binary_alpha": T_BINARY_ALPHA,
    "binary_cmp": T_BINARY_CMP,
    "binary_scalar": T_BINARY_SCALAR,
    "binary_scalar_alpha": T_BINARY_SCALAR_ALPHA,
    "binary_scalar_as_tensor": T_BINARY_SCALAR_AS_TENSOR,
    "binary_scalar_alpha_as_tensor": T_BINARY_SCALAR_ALPHA_AS_TENSOR,
    "binary_scalar_cmp": T_BINARY_SCALAR_CMP,
    "matmul": T_MATMUL,
    "matmul_out": T_MATMUL_OUT,
    "reduce_dims_dtype": T_REDUCE_DIMS_DTYPE,
    "reduce_all_dtype": T_REDUCE_ALL_DTYPE,
    "gelu": T_GELU,
    "softmax_fwd": T_SOFTMAX_FWD,
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_gcu.py -- DO NOT EDIT.
//
// topsaten kernels for the Enflame GCU backend, generated per-category. Each
// kernel wraps its aten tensors into topsatenTensors and issues one direct
// topsaten call via EXEC_TOPSATEN_CMD. Dispatchers are declared in
// generated/ops.h (shared with the CUDA codegen); here we only fill the
// Backend::kGcu slot.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/result_type.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../topsaten_common.h"

namespace at::native::flagos {

namespace gcu = at::native::flagos::gcu;

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_gcu.py -- DO NOT EDIT.
//
// m.impl() lines for the ops that have a topsaten kernel. Included by
// register.cc inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_GCU is
// set, in place of the full generated/register.inc list: an op registered on
// PrivateUse1 without a kernel behind it fails the dispatcher's
// "backend not registered" check, whereas an unregistered op simply reaches
// the cpu_fallback. So this file is exactly the GCU coverage set.

"""


def topsaten_name(op_base: str, override) -> str:
    if override:
        return "topsaten" + override
    pascal = "".join(w.capitalize() for w in op_base.lstrip("_").split("_") if w)
    return "topsaten" + pascal


def libtopsaten_path() -> Path:
    env = os.environ.get("TOPSATEN_LIB")
    if env:
        return Path(env)
    for cand in ("/usr/lib/libtopsaten.so", "/opt/tops/lib/libtopsaten.so"):
        if Path(cand).exists():
            return Path(cand)
    return Path("/usr/lib/libtopsaten.so")


def symbols(lib: Path):
    """topsaten entry points are C++ functions in `namespace topsaten`, so the
    mangled names must be demangled (nm -C) before matching."""
    if not lib.exists():
        print(f"[warn] {lib} not found; skipping symbol validation", file=sys.stderr)
        return None
    out = subprocess.run(
        ["nm", "-DC", "--defined-only", str(lib)], capture_output=True, text=True
    )
    return set(re.findall(r"topsaten::(topsaten\w+)", out.stdout))


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
        help="do not append covered ops to backends_gcu.conf",
    )
    args = ap.parse_args()

    syms = symbols(libtopsaten_path())
    wrappers = wrapper_map()

    bodies = []
    covered = []  # (op, tops, category)
    skipped = []  # (op, reason)

    for op, (cat, override) in OPS.items():
        if args.category != "all" and cat != args.category:
            continue
        if op in SKIP:
            skipped.append((op, "handwritten"))
            continue
        base = op.split(".")[0]
        tops = topsaten_name(base, override)
        if syms is not None and tops not in syms:
            skipped.append((op, f"{tops} not in {libtopsaten_path().name}"))
            continue
        if op not in wrappers:
            skipped.append((op, "no wrapper in generated/register.inc"))
            continue
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelGcu"  # SqrtFn -> SqrtKernelGcu
        bodies.append(
            CATEGORIES[cat].format(
                kernel=kernel,
                tops=tops,
                fn=fn,
                disp=disp,
                at_op=AT_OP_OVERRIDES.get(op, base),
                promote_integral="true" if base == "sum" else "false",
            )
        )
        covered.append((op, tops, cat))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    impls = "".join(
        f'  m.impl("{op}", {wrappers[op]});\n' for op, _, _ in sorted(covered)
    )
    OUT_INC.write_text(INC_HEADER + impls)

    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    print(f"[gen] {OUT_INC.relative_to(REPO)}  ({len(covered)} m.impl lines)")
    by_cat = {}
    for op, tops, cat in covered:
        by_cat.setdefault(cat, []).append((op, tops))
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if items:
            print(f"    [{cat}] {len(items)}")
            for op, tops in items:
                print(f"       + {op} -> {tops}")
    for op, why in skipped:
        print(f"       - {op} skipped ({why})")

    if not args.no_conf and covered:
        existing = CONF.read_text() if CONF.exists() else ""
        # Strip any prior codegen block so re-runs stay idempotent.
        marker = "\n# --- generated by codegen_gcu.py"
        if marker in existing:
            existing = existing[: existing.index(marker)].rstrip() + "\n"
        lines = []
        for op, _, _ in covered:
            if f"\n{op} = " not in ("\n" + existing):
                lines.append(f"{op} = gcu")
        new = existing.rstrip() + "\n"
        if lines:
            new += "\n# --- generated by codegen_gcu.py ---\n"
            new += "\n".join(lines) + "\n"
        CONF.write_text(new)
        print(f"[conf] wrote {len(lines)} generated op(s) to {CONF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
