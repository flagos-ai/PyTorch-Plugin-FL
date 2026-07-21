#!/usr/bin/env python3
"""
Codegen for torch_fl Ascend (aclnn) operators.

Unlike the CUDA codegen (scripts/codegen_ops.py), which emits a one-line
`at::op(args)` boxing body and relies on the dedicated CUDA dispatch key,
Ascend has no independent key to box into (torch_npu shares PrivateUse1 with
flagos). So every Ascend kernel must call CANN aclnn (libopapi.so) directly.

Because the aclnn call shape is uniform per *category*, this generator is
category-driven: a mapping table classifies each op, and a per-category
template emits the argument marshalling + output allocation + EXEC_ASCEND_CMD.

Categories implemented:
    unary               1 Tensor in -> 1 Tensor out (same shape/dtype)
                          aclnn<Name>(self, out)
    binary              2 Tensor in, broadcast, preserve dtype
                          aclnn<Name>(self, other, out)
    binary_alpha        binary + trailing Scalar alpha (sub.Tensor)
                          aclnn<Name>(self, other, alpha, out)
    binary_cmp          2 Tensor in, broadcast, bool out (eq/lt/logical_and..)
                          aclnn<Name>(self, other, out)
    binary_scalar_alpha Tensor + Scalar other + Scalar alpha (add/sub.Scalar)
                          aclnn<Name>s(self, other, alpha, out)
    binary_scalar_cmp   Tensor + Scalar, bool out (eq/lt.Scalar..)
                          aclnn<Name>(self, other, out)

Reuses:
  - scripts/codegen_ops.py:schema_to_cpp_name  (symbol names must match the
    dispatcher declarations already emitted into generated/ops.h)
  - the dispatchers/DECLARE_DISPATCHER already present in generated/ops.h
    (we only add the Backend::kAscend slot; we declare nothing new)

Generates:
  - csrc/aten/backends/ascend/generated/ascend_kernels.cc
  - appends newly-covered ops to torch_fl/backends_ascend.conf

Validation:
  - each derived aclnn<Name>/<Name>GetWorkspaceSize symbol must exist in
    libopapi.so, else the op is skipped with a warning.
  - handwritten ops (SKIP set) are never re-emitted (would double-register
    kAscend and crash at import).

Known limitations (documented, acceptable for the current op set):
  - binary/binary_alpha take the output dtype from `self`; ops with C++-level
    type promotion on mixed-dtype inputs (e.g. integer div -> float) are not
    modelled. Typical float workloads are unaffected.
  - No category may pass a by-value `float`/`double` argument to aclnn.
    EXEC_ASCEND_CMD calls the aclnn entry through a variadic `int(*)(...)`
    pointer, and on AArch64 a by-value float goes through varargs promotion
    (float->double, wrong register class) so aclnn reads a garbage value.
    Scalars must be marshalled as `aclScalar*`; int64/bool pass through fine.
    (This is why smooth_l1_loss, which has a `float beta`, is left long-tail.)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Reuse the authoritative symbol-naming from the CUDA codegen so the emitted
# REGISTER_IMPL_TO_DISPATCHER(FnType, dispatcher, ...) matches the
# DECLARE_DISPATCHER in generated/ops.h exactly (else the build won't link).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen_ops import schema_to_cpp_name  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_CC = REPO / "csrc/aten/backends/ascend/generated/ascend_kernels.cc"
CONF = REPO / "torch_fl/backends_ascend.conf"

# --------------------------------------------------------------------------
# Op registry: schema op name -> (category, aclnn-name override or None).
#
# Default aclnn name = "aclnn" + PascalCase(op base). A non-None override
# replaces that stem for irregular spellings (e.g. eq.Tensor -> aclnnEqTensor).
#
# Ops already handwritten in csrc/aten/backends/ascend/*.cc must NOT appear
# here (they already own the kAscend slot; double-register crashes at import).
# --------------------------------------------------------------------------
OPS = {
    # ---- unary: aclnn<Name>(self, out), out = self.shape/dtype ----
    "sqrt":         ("unary", None),
    "exp":          ("unary", None),
    "tanh":         ("unary", None),
    "sigmoid":      ("unary", None),
    "reciprocal":   ("unary", None),
    "log":          ("unary", None),
    "floor":        ("unary", None),
    "ceil":         ("unary", None),
    "erf":          ("unary", None),
    "erfc":         ("unary", None),
    "expm1":        ("unary", None),
    "log2":         ("unary", None),
    "log10":        ("unary", None),
    "log1p":        ("unary", None),
    "round":        ("unary", None),
    "trunc":        ("unary", None),
    "frac":         ("unary", None),
    "sign":         ("unary", None),
    "relu":         ("unary", None),
    "cosh":         ("unary", None),
    "sinh":         ("unary", None),
    "asin":         ("unary", None),
    "atan":         ("unary", None),
    "asinh":        ("unary", None),
    "acosh":        ("unary", None),
    "atanh":        ("unary", None),
    "logical_not":  ("unary", None),
    "bitwise_not":  ("unary", None),

    # ---- binary: aclnn<Name>(self, other, out), broadcast, preserve dtype ----
    "div.Tensor":         ("binary", "Div"),
    "pow.Tensor_Tensor":  ("binary", "PowTensorTensor"),
    "atan2":              ("binary", None),
    "maximum":            ("binary", None),
    "minimum":            ("binary", None),
    "bitwise_or.Tensor":  ("binary", "BitwiseOrTensor"),
    "bitwise_xor.Tensor": ("binary", "BitwiseXorTensor"),

    # ---- binary_alpha: aclnn<Name>(self, other, alpha, out) ----
    "sub.Tensor":         ("binary_alpha", "Sub"),

    # ---- binary_cmp: bool out ----
    "eq.Tensor":          ("binary_cmp", "EqTensor"),
    "ne.Tensor":          ("binary_cmp", "NeTensor"),
    "gt.Tensor":          ("binary_cmp", "GtTensor"),
    "lt.Tensor":          ("binary_cmp", "LtTensor"),
    "ge.Tensor":          ("binary_cmp", "GeTensor"),
    "logical_and":        ("binary_cmp", None),
    "logical_or":         ("binary_cmp", None),

    # ---- binary_scalar_alpha: aclnn<Name>s(self, other, alpha, out) ----
    "add.Scalar":         ("binary_scalar_alpha", "Adds"),
    "sub.Scalar":         ("binary_scalar_alpha", "Subs"),

    # ---- binary_scalar_cmp: bool out, aclnn<Name>(self, other, out) ----
    "eq.Scalar":          ("binary_scalar_cmp", "EqScalar"),
    "ne.Scalar":          ("binary_scalar_cmp", "NeScalar"),
    "gt.Scalar":          ("binary_scalar_cmp", "GtScalar"),
    "lt.Scalar":          ("binary_scalar_cmp", "LtScalar"),
    "ge.Scalar":          ("binary_scalar_cmp", "GeScalar"),
    "le.Scalar":          ("binary_scalar_cmp", "LeScalar"),

    # ---- reduce_dims: (Tensor, IntArrayRef dim, bool keepdim), same dtype ----
    "amax":               ("reduce_dims", None),
    "amin":               ("reduce_dims", None),

    # ---- reduce_dim_bool: (Tensor, int64_t dim, bool keepdim), bool out ----
    "any.dim":            ("reduce_dim_bool", "Any"),

    # ---- cumsum: (Tensor, int64_t dim, optional<ScalarType> dtype) ----
    "cumsum":             ("cumsum", None),

    # ---- cumprod: like cumsum but aclnn takes dim as aclScalar*, not int64_t ----
    "cumprod":            ("cumprod", None),

    # ---- unary_bool: (Tensor) -> bool out ----
    "isinf":              ("unary_bool", "IsInf"),

    # ---- unary_scalar: (Tensor, Scalar) -> same shape ----
    "leaky_relu":         ("unary_scalar", None),
    "clamp_min":          ("unary_scalar", None),
    "clamp_max":          ("unary_scalar", None),
    "fmod.Scalar":        ("unary_scalar", "FmodScalar"),

    # ---- unary_two_scalar: (Tensor, Scalar, Scalar) -> same shape ----
    "softplus":           ("unary_two_scalar", None),
    "threshold":          ("unary_two_scalar", None),

    # ---- unary_int: (Tensor, int64_t) -> same shape ----
    "tril":               ("unary_int", None),
    "triu":               ("unary_int", None),

    # ---- unary_dims: (Tensor, IntArrayRef) -> same shape ----
    "flip":               ("unary_dims", None),

    # ---- addcmul: (self, t1, t2, Scalar value) -> broadcast ----
    "addcmul":            ("addcmul", None),
    "addcdiv":            ("addcmul", None),

    # ---- binary (tensor-tensor, preserve dtype) additions ----
    "fmod.Tensor":        ("binary", "FmodTensor"),
    "floor_divide":       ("binary", None),
    "logical_xor":        ("binary_cmp", None),

    # ---- act_backward: (grad_output, output) -> grad_input ----
    "tanh_backward":      ("act_backward", None),
    "sigmoid_backward":   ("act_backward", None),

    # ---- threshold_backward: (grad_output, self, Scalar threshold) ----
    "threshold_backward": ("threshold_backward", None),

    # ---- pow_scalar_tensor: (Scalar self, Tensor exponent) ----
    "pow.Scalar":         ("pow_scalar_tensor", "PowScalarTensor"),

    # ---- reduce_max_dim: (Tensor, int64_t dim, bool keepdim) -> (values, indices) ----
    "max.dim":            ("reduce_max_dim", "MaxDim"),
    "min.dim":            ("reduce_max_dim", "MinDim"),

    # ---- more unary_scalar activations ----
    "celu":               ("unary_scalar", None),
    "softshrink":         ("unary_scalar", None),
    "hardshrink":         ("unary_scalar", None),

    # ---- more unary_two_scalar (min/max clip) ----
    "hardtanh":           ("unary_two_scalar", None),

    # ---- elu: (Tensor, alpha, scale, input_scale) ----
    "elu":                ("elu", None),

    # ---- loss: (self, target, reduction) -> scalar/elementwise ----
    "mse_loss":           ("loss", "MseLoss"),
    # smooth_l1_loss/l1_loss(*): smooth_l1 needs a by-value `float beta`. The
    # EXEC_ASCEND_CMD macro calls the aclnn entry through a variadic
    # `int(*)(...)` pointer; on AArch64 a by-value float passed through varargs
    # is promoted to double and lands in the wrong register class, so aclnn
    # reads beta as 0 (result collapses to pure L1). Left long-tail until the
    # macro grows a typed-call path for by-value floats.

    # ---- cummax/cummin: (Tensor, dim) -> tuple(values, indices) ----
    "cummax":             ("cummax_cummin", "Cummax"),
    "cummin":             ("cummax_cummin", "Cummin"),

    # ---- aminmax: (Tensor, optional dim, keepdim) -> tuple(min, max) ----
    "aminmax":            ("aminmax", "Aminmax"),

    # ---- prod: (Tensor, optional dtype) -> scalar ----
    "prod":               ("prod", "Prod"),

    # ---- gemm family (cubeMathType). addbmm reduces the batch dim (2D out) and
    #      addmv/addr have different arg order / no cubeMathType -> left long-tail.
    "addmm":              ("gemm_addmm", "Addmm"),
    "baddbmm":            ("gemm_baddbmm", "Baddbmm"),
    "mv":                 ("mv", "Mv"),
    "dot":                ("dot", "Dot"),
}

# Ops with a handwritten kAscend kernel — never regenerate (double-register).
# Kept as a guard even though none currently overlap OPS above.
SKIP = {
    "abs", "acos", "cos", "sin", "neg", "rsqrt", "silu",
    "add.Tensor", "mul.Tensor", "mul.Scalar", "div.Scalar",
    "pow.Tensor_Scalar", "le.Tensor", "bitwise_and.Tensor", "where.self",
}

# --------------------------------------------------------------------------
# Per-category kernel body templates. Placeholders: {kernel} {aclnn} {fn} {disp}
# --------------------------------------------------------------------------
T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shared broadcast+dtype prologue for tensor-tensor categories.
#
# `other` is coerced to self's DEVICE and dtype: the aten .Tensor overloads
# (add.Tensor/sub.Tensor/...) are what the Python operators lower a scalar
# argument to (e.g. `x - 3.0` -> aten::sub.Tensor with a CPU scalar tensor),
# so `other` may live on CPU. Building an aclTensor over CPU storage as if it
# were NPU memory yields garbage/nan, so we must migrate it to self's device
# first (mirrors the handwritten add.cc). Both operands are then expanded +
# materialized to the broadcast shape so aclnn (which does not always
# broadcast) sees matching ND-contiguous inputs. All steps are no-ops when
# device/dtype/shape already match.
_BINARY_PROLOGUE = """\
  namespace ascend = at::native::flagos::ascend;
  auto result_dtype = self.scalar_type();
  auto other_c = other.is_privateuseone()
      ? (other.scalar_type() == result_dtype ? other : other.to(result_dtype))
      : other.to(self.options());
  auto out_shape = at::infer_size(self.sizes(), other_c.sizes());
  auto self_b = self.expand(out_shape).contiguous();
  auto other_b = other_c.expand(out_shape).contiguous();
"""

T_BINARY = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
""" + _BINARY_PROLOGUE + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY_ALPHA = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
""" + _BINARY_PROLOGUE + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclScalarWrapper acl_alpha(alpha, result_dtype);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_alpha.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY_CMP = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
""" + _BINARY_PROLOGUE + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY_SCALAR_ALPHA = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_alpha.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY_SCALAR_CMP = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shared dim-normalization + reduced-shape prologue for reduce categories.
# Mirrors the handwritten sum.cc: negative dims are wrapped to [0,ndim); an
# empty dim list means "reduce all"; the output shape drops (or, with keepdim,
# sets to 1) each reduced dim. Dims are erased high-to-low so earlier erases do
# not shift later indices.
_REDUCE_DIMS_PROLOGUE = """\
  namespace ascend = at::native::flagos::ascend;
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  if (!dim.empty()) {{
    for (int64_t d : dim) norm_dims.push_back(d < 0 ? d + ndim : d);
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
"""

# reduce_dims: (Tensor, IntArrayRef dim, bool keepdim) -> reduced, same dtype.
#   aclnn<Name>(self, dim, keepdim, out)   e.g. amax/amin
T_REDUCE_DIMS = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {{
""" + _REDUCE_DIMS_PROLOGUE + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(norm_dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_dim_bool: (Tensor, int64_t dim, bool keepdim) -> bool out.
#   aclnn<Name>(self, dim_list, keepdim, out)   e.g. any.dim
# aclnn takes a dim *list*, so the single dim is wrapped into a one-element vec.
T_REDUCE_DIM_BOOL = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool keepdim) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  std::vector<int64_t> dims{{d}};
  auto out_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cumsum: (Tensor, int64_t dim, optional<ScalarType> dtype) -> same shape.
#   aclnn<Name>(self, dim, dtype, out)
T_CUMSUM = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_dtype = dtype.value_or(self.scalar_type());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cumprod: like cumsum, but aclnnCumprod takes dim as an aclScalar* (int64), not
# a plain int64_t. Otherwise identical: (Tensor, int64 dim, optional dtype).
T_CUMPROD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_dtype = dtype.value_or(self.scalar_type());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_dim(at::Scalar(d), at::kLong);
  ascend::AclTensorWrapper acl_out(out);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_bool: (Tensor) -> bool out, same shape.  aclnn<Name>(self, out)
#   e.g. isinf (always bool regardless of input dtype)
T_UNARY_BOOL = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_scalar: (Tensor, Scalar) -> same shape/dtype.
#   aclnn<Name>(self, scalar, out)   e.g. leaky_relu/clamp_min/clamp_max/fmod.Scalar
# The scalar is packed at self's dtype so aclnn sees matching operand types.
T_UNARY_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& s) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_s(s, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_s.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_two_scalar: (Tensor, Scalar, Scalar) -> same shape/dtype.
#   aclnn<Name>(self, s1, s2, out)   e.g. softplus(beta,threshold)/threshold(threshold,value)
T_UNARY_TWO_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& s1, const at::Scalar& s2) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_s1(s1, self.scalar_type());
  ascend::AclScalarWrapper acl_s2(s2, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_s1.get(), acl_s2.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_int: (Tensor, int64_t) -> same shape/dtype.  aclnn<Name>(self, i, out)
#   e.g. tril/triu (diagonal offset)
T_UNARY_INT = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t i) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), i, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_dims: (Tensor, IntArrayRef dims) -> same shape/dtype.
#   aclnn<Name>(self, dims, out)   e.g. flip
T_UNARY_DIMS = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef dims) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  std::vector<int64_t> dims_v(dims.begin(), dims.end());
  ascend::AclIntArrayWrapper acl_dims(dims_v);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dims.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# addcmul/addcdiv: (self, t1, t2, Scalar value) -> broadcast(self,t1,t2), self dtype.
#   aclnn<Name>(self, t1, t2, value, out)
# t1/t2 are migrated to self's device+dtype (same rationale as _BINARY_PROLOGUE:
# a CPU-resident operand read as NPU storage would produce garbage).
T_ADDCMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto opts = self.options();
  auto t1 = tensor1.is_privateuseone() ? tensor1.to(opts.dtype(tensor1.scalar_type())) : tensor1.to(opts);
  auto t2 = tensor2.is_privateuseone() ? tensor2.to(opts.dtype(tensor2.scalar_type())) : tensor2.to(opts);
  auto out_shape = at::infer_size(at::infer_size(self.sizes(), t1.sizes()), t2.sizes());
  auto self_b = self.expand(out_shape).contiguous();
  auto t1_b = t1.expand(out_shape).contiguous();
  auto t2_b = t2.expand(out_shape).contiguous();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, opts);

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_t1(t1_b);
  ascend::AclTensorWrapper acl_t2(t2_b);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_t1.get(), acl_t2.get(), acl_value.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# act_backward: (grad_output, output) -> grad_input, output shape/dtype.
#   aclnn<Name>(gradOutput, output, gradInput)   e.g. tanh_backward/sigmoid_backward
T_ACT_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& output) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      output.sizes(), output.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_output(output);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_output.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# threshold_backward: (grad_output, self, Scalar threshold) -> grad_input.
#   aclnn<Name>(gradOutput, self, threshold, gradInput)
T_THRESHOLD_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_threshold(threshold, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_threshold.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# pow_scalar_tensor: (Scalar self, Tensor exponent) -> exponent shape.
#   aclnn<Name>(selfScalar, exponent, out)   e.g. pow.Scalar
T_POW_SCALAR_TENSOR = """\
at::Tensor {kernel}(const at::Scalar& self, const at::Tensor& exponent) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      exponent.sizes(), exponent.options());

  ascend::AclScalarWrapper acl_self(self, exponent.scalar_type());
  ascend::AclTensorWrapper acl_exp(exponent);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_exp.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_max_dim: (Tensor, int64_t dim, bool keepdim) -> tuple(values, indices).
#   aclnn<Name>(self, dim, keepdim, valuesOut, indicesOut)   e.g. max.dim/min.dim
# values keep self dtype; indices are int64.
T_REDUCE_MAX_DIM = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, int64_t dim, bool keepdim) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);

  auto values = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_values(values);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, keepdim, acl_values.get(), acl_indices.get());
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# elu: (Tensor, Scalar alpha, Scalar scale, Scalar input_scale) -> same shape.
#   aclnn<Name>(self, alpha, scale, inputScale, out)
T_ELU = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclScalarWrapper acl_scale(scale, self.scalar_type());
  ascend::AclScalarWrapper acl_input_scale(input_scale, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_alpha.get(), acl_scale.get(), acl_input_scale.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# loss: (self, target, int64_t reduction) -> reduction==None(0) elementwise,
# Mean(1)/Sum(2) scalar.  aclnn<Name>(self, target, reduction, out)
# e.g. mse_loss / l1_loss. target is a genuine on-device tensor (no CPU-scalar
# lowering as with the binary categories), so no device coercion is needed.
T_LOSS = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar for Mean/Sum
  if (reduction == 0) out_shape = self.sizes().vec();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_target.get(), reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cummax/cummin: (Tensor, int64_t dim) -> tuple(values, indices), same shape.
#   aclnn<Name>(self, dim, valuesOut, indicesOut)
T_CUMMAX_CUMMIN = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, int64_t dim) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto values = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_values(values);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_values.get(), acl_indices.get());
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# aminmax: (Tensor, optional<int64_t> dim, bool keepdim) -> tuple(min, max).
#   aclnn<Name>(self, dim_list, keepDim, minOut, maxOut)
# nullopt dim reduces all dims (scalar out); a given dim reduces that one.
T_AMINMAX = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, ::std::optional<int64_t> dim, bool keepdim) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> dims;
  std::vector<int64_t> out_shape;
  if (dim.has_value()) {{
    int64_t d = dim.value() < 0 ? dim.value() + self.dim() : dim.value();
    dims.push_back(d);
    out_shape = self.sizes().vec();
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
  }} else {{
    for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
  }}
  auto min_out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());
  auto max_out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_min(min_out);
  ascend::AclTensorWrapper acl_max(max_out);
  ascend::AclIntArrayWrapper acl_dim(dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_min.get(), acl_max.get());
  return std::make_tuple(min_out, max_out);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# prod: (Tensor, optional<ScalarType> dtype) -> scalar.
#   aclnn<Name>(self, dtype, out)
T_PROD = """\
at::Tensor {kernel}(const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = dtype.value_or(self.scalar_type());
  std::vector<int64_t> out_shape;   // scalar
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_addmm: (self, mat1, mat2, Scalar beta, Scalar alpha) -> (m1.rows, m2.cols).
#   aclnn<Name>(self, mat1, mat2, beta, alpha, out, cubeMathType)   e.g. addmm
# self broadcasts into the matmul result; aclnn handles the broadcast.
T_GEMM_ADDMM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{mat1.size(0), mat2.size(1)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat1(mat1);
  ascend::AclTensorWrapper acl_mat2(mat2);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_mat1.get(), acl_mat2.get(), acl_beta.get(), acl_alpha.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_baddbmm: batched addmm -> (b, b1.rows, b2.cols).
#   aclnn<Name>(self, batch1, batch2, beta, alpha, out, cubeMathType)
T_GEMM_BADDBMM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{batch1.size(0), batch1.size(1), batch2.size(2)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_batch1(batch1);
  ascend::AclTensorWrapper acl_batch2(batch2);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_batch1.get(), acl_batch2.get(), acl_beta.get(), acl_alpha.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# mv: (self (n,m), vec (m,)) -> (n,).  aclnn<Name>(self, vec, out, cubeMathType)
T_MV = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& vec) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{self.size(0)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_vec(vec);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_vec.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# dot: (self, tensor) both 1-D -> scalar.  aclnn<Name>(self, tensor, out)
T_DOT = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& tensor) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_tensor(tensor);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_tensor.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

CATEGORIES = {
    "unary":               T_UNARY,
    "binary":              T_BINARY,
    "binary_alpha":        T_BINARY_ALPHA,
    "binary_cmp":          T_BINARY_CMP,
    "binary_scalar_alpha": T_BINARY_SCALAR_ALPHA,
    "binary_scalar_cmp":   T_BINARY_SCALAR_CMP,
    "reduce_dims":         T_REDUCE_DIMS,
    "reduce_dim_bool":     T_REDUCE_DIM_BOOL,
    "cumsum":              T_CUMSUM,
    "cumprod":             T_CUMPROD,
    "unary_bool":          T_UNARY_BOOL,
    "unary_scalar":        T_UNARY_SCALAR,
    "unary_two_scalar":    T_UNARY_TWO_SCALAR,
    "unary_int":           T_UNARY_INT,
    "unary_dims":          T_UNARY_DIMS,
    "addcmul":             T_ADDCMUL,
    "act_backward":        T_ACT_BACKWARD,
    "threshold_backward":  T_THRESHOLD_BACKWARD,
    "pow_scalar_tensor":   T_POW_SCALAR_TENSOR,
    "reduce_max_dim":      T_REDUCE_MAX_DIM,
    "elu":                 T_ELU,
    "loss":                T_LOSS,
    "cummax_cummin":       T_CUMMAX_CUMMIN,
    "aminmax":             T_AMINMAX,
    "prod":                T_PROD,
    "gemm_addmm":          T_GEMM_ADDMM,
    "gemm_baddbmm":        T_GEMM_BADDBMM,
    "mv":                  T_MV,
    "dot":                 T_DOT,
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_ascend.py -- DO NOT EDIT.
//
// aclnn kernels for the Ascend backend, generated per-category. Each kernel
// marshals aten tensors into aclTensors and issues the two-phase aclnn call
// via EXEC_ASCEND_CMD. Dispatchers are declared in generated/ops.h (shared
// with the CUDA codegen); here we only fill the Backend::kAscend slot.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <vector>
#include "../op_preparation.h"
#include "../op_api_common.h"

namespace at::native::flagos {

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"


def aclnn_name(op_base: str, override) -> str:
    if override:
        return "aclnn" + override
    pascal = "".join(w.capitalize() for w in op_base.split("_") if w)
    return "aclnn" + pascal


def libopapi_path() -> Path:
    ah = os.environ.get("ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
    return Path(ah) / "lib64" / "libopapi.so"


def symbols(lib: Path):
    if not lib.exists():
        print(f"[warn] {lib} not found; skipping symbol validation", file=sys.stderr)
        return None
    out = subprocess.run(["nm", "-D", str(lib)], capture_output=True, text=True)
    syms = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts:
            syms.add(parts[-1])
    return syms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="all",
                    choices=["all"] + list(CATEGORIES),
                    help="restrict generation to one category (default: all)")
    ap.add_argument("--no-conf", action="store_true",
                    help="do not append covered ops to backends_ascend.conf")
    args = ap.parse_args()

    syms = symbols(libopapi_path())

    bodies = []
    covered = []   # (op, aclnn, category)
    skipped = []   # (op, reason)

    for op, (cat, override) in OPS.items():
        if args.category != "all" and cat != args.category:
            continue
        if op in SKIP:
            skipped.append((op, "handwritten"))
            continue
        base = op.split(".")[0]
        acl = aclnn_name(base, override)
        if syms is not None:
            if (acl not in syms) or (acl + "GetWorkspaceSize" not in syms):
                skipped.append((op, f"{acl} not in libopapi.so"))
                continue
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelAscend"  # SqrtFn -> SqrtKernelAscend
        bodies.append(CATEGORIES[cat].format(
            kernel=kernel, aclnn=acl, fn=fn, disp=disp))
        covered.append((op, acl, cat))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    # Report grouped by category.
    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    by_cat = {}
    for op, acl, cat in covered:
        by_cat.setdefault(cat, []).append((op, acl))
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if items:
            print(f"    [{cat}] {len(items)}")
            for op, acl in items:
                print(f"       + {op} -> {acl}")
    for op, why in skipped:
        print(f"       - {op} skipped ({why})")

    if not args.no_conf and covered:
        existing = CONF.read_text() if CONF.exists() else ""
        # Strip any prior codegen block so re-runs stay idempotent.
        marker = "\n# --- generated by codegen_ascend.py"
        if marker in existing:
            existing = existing[:existing.index(marker)].rstrip() + "\n"
        lines = []
        for op, _, _ in covered:
            if f"\n{op} = " not in ("\n" + existing):
                lines.append(f"{op} = ascend")
        new = existing.rstrip() + "\n"
        if lines:
            new += "\n# --- generated by codegen_ascend.py ---\n"
            new += "\n".join(lines) + "\n"
        CONF.write_text(new)
        print(f"[conf] wrote {len(lines)} generated op(s) to {CONF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
