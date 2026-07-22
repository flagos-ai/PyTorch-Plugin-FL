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
    # migrated from handwritten seed kernels (bodies were identical to T_UNARY):
    "abs":          ("unary", None),
    "acos":         ("unary", None),
    "cos":          ("unary", None),
    "sin":          ("unary", None),
    "neg":          ("unary", None),
    "rsqrt":        ("unary", None),
    "silu":         ("unary", None),

    # ---- binary: aclnn<Name>(self, other, out), broadcast, preserve dtype ----
    "div.Tensor":         ("binary", "Div"),
    # migrated from handwritten seeds:
    "mul.Tensor":         ("binary", "Mul"),
    "bitwise_and.Tensor": ("binary", "BitwiseAndTensor"),
    "pow.Tensor_Tensor":  ("binary", "PowTensorTensor"),
    "atan2":              ("binary", None),
    "maximum":            ("binary", None),
    "minimum":            ("binary", None),
    "bitwise_or.Tensor":  ("binary", "BitwiseOrTensor"),
    "bitwise_xor.Tensor": ("binary", "BitwiseXorTensor"),

    # ---- binary_alpha: aclnn<Name>(self, other, alpha, out) ----
    "sub.Tensor":         ("binary_alpha", "Sub"),
    "add.Tensor":         ("binary_alpha", "Add"),   # migrated seed

    # ---- binary_scalar: aclnn<Name>(self, scalar, out), no alpha (migrated seeds) ----
    "mul.Scalar":         ("binary_scalar", "Muls"),
    "div.Scalar":         ("binary_scalar", "Divs"),

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
    "pow.Tensor_Scalar":  ("unary_scalar", "PowTensorScalar"),   # migrated seed

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

    # ---- gemm family (cubeMathType) ----
    "addmm":              ("gemm_addmm", "Addmm"),
    "baddbmm":            ("gemm_baddbmm", "Baddbmm"),
    "mv":                 ("mv", "Mv"),
    "dot":                ("dot", "Dot"),
    # addmv: mat(n,m) x vec(m) -> (n,); aclnn arg order is (self,mat,vec,ALPHA,BETA).
    "addmv":              ("gemm_addmv", "Addmv"),
    # addr: outer(vec1(n), vec2(m)) -> (n,m); no cubeMathType.
    "addr":               ("gemm_addr", "Addr"),
    # NOTE addbmm left out: hf32 cube accumulation over the batch dim inflates
    #   rel-err to ~1e-2 (single addmm is ~1e-4).

    # ---- CNN training closure: pool bwd + batch norm ----
    "max_pool2d_with_indices_backward": ("max_pool2d_indices_backward", "MaxPool2dWithIndicesBackward"),
    "native_batch_norm":          ("native_batch_norm", "BatchNorm"),
    "native_batch_norm_backward": ("native_batch_norm_backward", "BatchNormBackward"),
    "avg_pool2d_backward":            ("avg_pool2d_backward", "AvgPool2dBackward"),
    "_adaptive_avg_pool2d_backward":  ("adaptive_avg_pool2d_backward", "AdaptiveAvgPool2dBackward"),
    "native_layer_norm_backward":     ("native_layer_norm_backward", "LayerNormBackward"),
    "native_group_norm_backward":     ("native_group_norm_backward", "GroupNormBackward"),

    # ---- Transformer indexing / masking (aclnn masked_fill is INPLACE-only) ----
    "masked_fill.Scalar":  ("masked_fill_scalar", "InplaceMaskedFillScalar"),
    "masked_fill.Tensor":  ("masked_fill_tensor", "InplaceMaskedFillTensor"),
    "gather":              ("gather", "Gather"),
    "index_select":        ("index_select", "IndexSelect"),

    # ---- in-place zero/fill (aclnn Inplace* ops); device-side, no h2d.
    #   Factory ops (zeros/ones_like/new_ones/scalar_tensor) call these. ----
    "zero_":               ("inplace_zero", "InplaceZero"),
    "fill_.Scalar":        ("inplace_fill_scalar", "InplaceFillScalar"),
    "fill_.Tensor":        ("inplace_fill_tensor", "InplaceFillTensor"),

    # ---- embedding + pad (single-aclnn-call, migrated from handwritten) ----
    "embedding":               ("embedding", "Embedding"),
    "embedding_dense_backward": ("embedding_dense_backward", "EmbeddingDenseBackward"),
    "constant_pad_nd":         ("constant_pad_nd", "ConstantPadNd"),

    # ---- BCE loss family: optional weight, int reduction (0=none/1=mean/2=sum) ----
    "binary_cross_entropy":              ("bce", "BinaryCrossEntropy"),
    "binary_cross_entropy_backward":     ("bce_backward", "BinaryCrossEntropyBackward"),
    "binary_cross_entropy_with_logits":  ("bce_logits", "BinaryCrossEntropyWithLogits"),

    # ---- norm family: tuple(out, mean, rstd), optional weight/bias ----
    "native_layer_norm":  ("layer_norm", "LayerNorm"),
    "native_group_norm":  ("group_norm", "GroupNorm"),

    # ---- gelu / softmax family (transformer backbone, fwd + bwd) ----
    # gelu: v1 aclnnGelu hardcodes tanh; use V2 (int64 approximate 0=none/1=tanh)
    # to honor PyTorch's default approximate="none" (erf form).
    "gelu":                       ("gelu", "GeluV2"),
    "gelu_backward":              ("gelu_backward", "GeluBackwardV2"),
    # _log_softmax mirrors handwritten softmax.cc (aclnnLogSoftmax(self,dim,out)).
    "_log_softmax":               ("log_softmax", "LogSoftmax"),
    # backward: aclnn names lack the aten "_data" suffix.
    "_softmax_backward_data":     ("softmax_backward", "SoftmaxBackward"),
    "_log_softmax_backward_data": ("softmax_backward", "LogSoftmaxBackward"),

    # ---- migrated seeds needing dedicated categories ----
    "silu_backward":      ("act_backward_self", "SiluBackward"),
    "where.self":         ("where", "SWhere"),
    "_softmax":           ("softmax_fwd", "Softmax"),
    "all":                ("reduce_all", "All"),
    "sum.dim_IntList":    ("reduce_sum_dtype", "ReduceSum"),
    "mean.dim":           ("reduce_mean_dtype", "MeanV2"),

    # ---- conv/pool family (each carries an output-shape formula) ----
    "_adaptive_avg_pool2d":     ("adaptive_avg_pool2d", "AdaptiveAvgPool2d"),
    "avg_pool2d":               ("avg_pool2d", "AvgPool2d"),
    "max_pool2d_with_indices":  ("max_pool2d_indices", "MaxPool2dWithIndices"),
    "convolution":              ("convolution", "Convolution"),
    "convolution_backward":     ("convolution_backward", "ConvolutionBackward"),
}

# Ops with a handwritten kAscend kernel — never regenerate (double-register).
# Kept as a guard even though none currently overlap OPS above.
SKIP = {
    # le.Tensor stays handwritten: aclnnLe symbol is absent, needs runtime
    # multi-version probing (aclnnLe / aclnnLeTensor / aclnnLessEqual).
    "le.Tensor",
    # mm/bmm stay handwritten: they also register out-variants (Mm/BmmOutFn)
    # that codegen does not emit.
    "mm", "bmm",
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

# gemm_addmv: (self, mat(n,m), vec(m), beta, alpha) -> (n,). NOTE the aclnn arg
#   order is (self, mat, vec, ALPHA, BETA) -- alpha before beta, unlike addmm.
T_GEMM_ADDMV = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{mat.size(0)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat(mat);
  ascend::AclTensorWrapper acl_vec(vec);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_mat.get(), acl_vec.get(), acl_alpha.get(), acl_beta.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_addr: (self, vec1(n), vec2(m), beta, alpha) -> (n,m) outer product.
#   aclnnAddr(self, vec1, vec2, beta, alpha, out) -- no cubeMathType.
T_GEMM_ADDR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& vec1, const at::Tensor& vec2, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape = {{vec1.size(0), vec2.size(0)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_vec1(vec1);
  ascend::AclTensorWrapper acl_vec2(vec2);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_vec1.get(), acl_vec2.get(), acl_beta.get(), acl_alpha.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# bce: (self, target, optional weight, int reduction) -> Tensor.
#   aclnnBinaryCrossEntropy(self, target, weight, reduction, out).
#   reduction 0=none -> out=self.shape; 1=mean/2=sum -> scalar. weight may be
#   undefined -> AclTensorWrapper yields nullptr (aclnn treats as absent).
T_BCE = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& target, const ::std::optional<at::Tensor>& weight, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar for mean/sum
  if (reduction == 0) out_shape = self.sizes().vec();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_target.get(), acl_weight.get(), reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# bce_backward: (grad_output, self, target, optional weight, int reduction).
#   aclnnBinaryCrossEntropyBackward(grad, self, target, weight, reduction, out).
#   grad_input = self.shape.
T_BCE_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, const ::std::optional<at::Tensor>& weight, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(self.sizes(), self.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_self.get(), acl_target.get(), acl_weight.get(), reduction, acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# bce_logits: (self, target, optional weight, optional pos_weight, reduction).
#   aclnnBinaryCrossEntropyWithLogits(self, target, weight, posWeight, reduction, out).
T_BCE_LOGITS = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& target, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& pos_weight, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar for mean/sum
  if (reduction == 0) out_shape = self.sizes().vec();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor pos_weight_t = pos_weight.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_pos_weight(pos_weight_t);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_target.get(), acl_weight.get(), acl_pos_weight.get(), reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_layer_norm: (input, IntArrayRef normalized_shape, optional weight,
#   optional bias, double eps) -> tuple(out, mean, rstd).
#   aclnn<Name>(input, normShape, weight, bias, eps, out, meanOut, rstdOut)
# out = input shape; mean/rstd = input.shape[:begin_axis] + 1s, where
# begin_axis = input.dim() - normalized_shape.size(). weight/bias may be
# undefined -> AclTensorWrapper yields nullptr (aclnn treats as absent).
T_LAYER_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, double eps) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t begin_axis = input.dim() - static_cast<int64_t>(normalized_shape.size());
  auto stat_shape = input.sizes().vec();
  for (int64_t i = begin_axis; i < input.dim(); ++i) stat_shape[i] = 1;

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto mean = ascend::OpPreparation::apply_tensor_without_format(
      stat_shape, input.options());
  auto rstd = ascend::OpPreparation::apply_tensor_without_format(
      stat_shape, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_input(input);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);

  std::vector<int64_t> ns(normalized_shape.begin(), normalized_shape.end());
  ascend::AclIntArrayWrapper acl_ns(ns);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_ns.get(), acl_weight.get(), acl_bias.get(), eps, acl_out.get(), acl_mean.get(), acl_rstd.get());
  return std::make_tuple(out, mean, rstd);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_group_norm: (input, optional weight, optional bias, int64 N, int64 C,
#   int64 HxW, int64 group, double eps) -> tuple(out, mean, rstd).
#   aclnn<Name>(self, gamma, beta, N, C, HxW, group, eps, out, meanOut, rstdOut)
# out = input shape; mean/rstd = (N, group).
T_GROUP_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& input, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto mean = ascend::OpPreparation::apply_tensor_without_format(
      {{N, group}}, input.options());
  auto rstd = ascend::OpPreparation::apply_tensor_without_format(
      {{N, group}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_input(input);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_weight.get(), acl_bias.get(), N, C, HxW, group, eps, acl_out.get(), acl_mean.get(), acl_rstd.get());
  return std::make_tuple(out, mean, rstd);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gelu: (self, c10::string_view approximate) -> Tensor. NOTE: aclnnGelu (v1)
# hardcodes the *tanh* approximation, but PyTorch's default is approximate=
# "none" (erf form, used by qwen3 et al). So we use aclnnGeluV2, whose
# int64_t approximate selects 0="none"/1="tanh" (int is varargs-safe). out =
# self shape/dtype.
T_GELU = """\
at::Tensor {kernel}(const at::Tensor& self, c10::string_view approximate) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t approx = (approximate == "tanh") ? 1 : 0;
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
      "gelu: unsupported approximate='", approximate, "'");
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), approx, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gelu_backward: (grad_output, self, approximate) -> Tensor. aclnnGeluBackwardV2
# takes the approximation as a `char*` string ("none"/"tanh"); a pointer is
# varargs-safe. grad_input = self shape. (v1 aclnnGeluBackward is tanh-only,
# same mismatch as forward.)
T_GELU_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
      "gelu_backward: unsupported approximate='", approximate, "'");
  std::string approx_str(approximate);
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_self.get(), approx_str.data(), acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _log_softmax: (self, int64 dim, bool half_to_float) -> Tensor. Mirrors the
# handwritten softmax.cc: aclnn<Name>(self, dim, out); half_to_float promotes
# the output dtype to float. out = self shape.
T_LOG_SOFTMAX = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), dim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _softmax_backward_data / _log_softmax_backward_data:
#   (grad_output, output, int64 dim, at::ScalarType input_dtype) -> Tensor.
#   aclnn<Name>(gradOutput, output, dim, gradInput). grad_input = grad_output
#   shape, dtype = input_dtype (the dtype the forward input had).
T_SOFTMAX_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      grad_output.sizes(), grad_output.options().dtype(input_dtype));

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_output(output);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_output.get(), dim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# binary_scalar: (Tensor, Scalar) -> same shape/dtype. Scalar op, NO alpha.
#   aclnn<Name>(self, scalar, out)   e.g. mul.Scalar->aclnnMuls, div.Scalar->aclnnDivs
# (aclnn headers for Muls/Divs are absent but the symbols exist; arg marshaling
#  confirmed from the handwritten mul_scalar.cc / div_scalar.cc.)
T_BINARY_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# act_backward_self: (grad_output, self) -> grad_input. Differs from act_backward
#   (grad, output) in taking `self` as the second tensor. e.g. silu_backward.
#   aclnn<Name>(gradOutput, self, gradInput)
T_ACT_BACKWARD_SELF = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# where: where.self(cond, self, other) -> broadcast(cond, self, other), self dtype.
#   aclnn<Name>(condition, self, other, out). aclnn does not broadcast, so all
#   three operands are expanded+contiguous to the common shape.
T_WHERE = """\
at::Tensor {kernel}(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = at::infer_size(self.sizes(), other.sizes());
  out_shape = at::infer_size(condition.sizes(), out_shape);

  auto cond_b = condition.expand(out_shape).contiguous();
  auto self_b = self.expand(out_shape).contiguous();
  auto other_b = other.expand(out_shape).contiguous();

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_cond(cond_b);
  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_cond.get(), acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# softmax_fwd: _softmax(self, int64 dim, bool half_to_float) -> same shape.
#   aclnn<Name>(self, dim, out). half_to_float promotes the output dtype to float.
T_SOFTMAX_FWD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), dim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_all: all(self) -> bool scalar over ALL elements. aclnnAll reduces along
#   a dim list, so flatten to 1-D and reduce dim=0 to a 0-d bool out.
#   aclnn<Name>(self_flat, dim_list, keepdim=false, out)
T_REDUCE_ALL = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto input = self.contiguous().reshape({{-1}});
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {{}}, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(input);
  ascend::AclTensorWrapper acl_out(out);

  int64_t dim_val = 0;
  std::vector<int64_t> dims{{dim_val}};
  ascend::AclIntArrayWrapper acl_dim(dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), false, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shared prologue for the dtype-aware reduce categories (sum.dim_IntList /
# mean.dim). OptionalIntArrayRef dim (None/empty = reduce all) + optional dtype.
# Mirrors the handwritten sum.cc dim-normalization and reduced-shape logic.
_REDUCE_DTYPE_PROLOGUE = """\
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = dtype.has_value() ? dtype.value() : self.scalar_type();
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
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(out_dtype));
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(norm_dims);
"""

# reduce_sum_dtype: sum.dim_IntList(self, int[]? dim, keepdim, ScalarType? dtype).
#   aclnnReduceSum(self, dims, keepdim, aclDataType, out)
T_REDUCE_SUM_DTYPE = """\
at::Tensor {kernel}(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {{
""" + _REDUCE_DTYPE_PROLOGUE + """\
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_mean_dtype: mean.dim(self, int[]? dim, keepdim, ScalarType? dtype).
#   aclnnMeanV2(self, dims, keepdim, int32 dtype, out)  -- MeanV2 for CANN 8.5.
T_REDUCE_MEAN_DTYPE = """\
at::Tensor {kernel}(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {{
""" + _REDUCE_DTYPE_PROLOGUE + """\
  auto acl_dtype = static_cast<int32_t>(ascend::ToAclDataType(out_dtype));

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# ==========================================================================
# conv / pool family. These need an explicit output-shape formula (aclnn wants
# the output pre-allocated), so each carries a small shape helper in its body.
# Shared pooling out-dim formula (matches PyTorch): for ceil_mode the division
# rounds up, else down; a ceil-mode start beyond padding is clamped back.
# ==========================================================================
_POOL_OUT_DIM = """\
  auto pool_out_dim = [](int64_t in, int64_t k, int64_t s, int64_t p,
                         int64_t d, bool ceil) -> int64_t {{
    int64_t num = in + 2 * p - d * (k - 1) - 1;
    int64_t out = (ceil ? (num + s - 1) / s : num / s) + 1;
    if (ceil && (out - 1) * s >= in + p) out -= 1;   // last window all-padding
    return out;
  }};
"""

# adaptive_avg_pool2d: (self, SymInt[2] output_size) -> Tensor.
#   out shape = self leading dims + output_size. aclnn<Name>(self, outSize, out).
T_ADAPTIVE_AVG_POOL2D = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef output_size) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = self.sizes().vec();
  int64_t r = out_shape.size();
  out_shape[r - 2] = output_size[0];
  out_shape[r - 1] = output_size[1];

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  // aclnn pooling rejects ND 4-D; tag as NCHW.
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclIntArrayWrapper acl_osize(output_size);
  ascend::AclTensorWrapper acl_out(out, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_osize.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# avg_pool2d: (self, kernel_size, stride, padding, ceil_mode, count_include_pad,
#   divisor_override) -> Tensor. stride defaults to kernel_size when empty.
#   aclnnAvgPool2d(self, k, s, p, ceil, countPad, divOverride, cubeMathType, out).
T_AVG_POOL2D = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());
""" + _POOL_OUT_DIM + """\
  auto out_shape = self.sizes().vec();
  int64_t r = out_shape.size();
  out_shape[r - 2] = pool_out_dim(self.size(r - 2), k[0], s[0], p[0], 1, ceil_mode);
  out_shape[r - 1] = pool_out_dim(self.size(r - 1), k[1], s[1], p[1], 1, ceil_mode);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  int64_t div_override = divisor_override.value_or(0);
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclTensorWrapper acl_out(out, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_k.get(), acl_s.get(), acl_p.get(),
      ceil_mode, count_include_pad, div_override, (int8_t)0, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# max_pool2d_with_indices: (self, k, stride, padding, dilation, ceil_mode) ->
#   tuple(out, int64 indices). stride defaults to kernel_size when empty.
#   aclnn<Name>(self, k, s, p, dil, ceil, out, indices).
T_MAX_POOL2D_INDICES = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());
  std::vector<int64_t> dil(dilation.begin(), dilation.end());
""" + _POOL_OUT_DIM + """\
  auto out_shape = self.sizes().vec();
  int64_t r = out_shape.size();
  out_shape[r - 2] = pool_out_dim(self.size(r - 2), k[0], s[0], p[0], dil[0], ceil_mode);
  out_shape[r - 1] = pool_out_dim(self.size(r - 1), k[1], s[1], p[1], dil[1], ceil_mode);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclIntArrayWrapper acl_dil(dil);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_k.get(), acl_s.get(), acl_p.get(),
      acl_dil.get(), ceil_mode, acl_out.get(), acl_indices.get());
  return std::make_tuple(out, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# masked_fill.Scalar: (self, mask, value) -> Tensor (= self broadcast mask).
#   aclnn only ships the INPLACE variant (aclnnInplaceMaskedFillScalar), so
#   clone self (broadcast to mask if needed) and fill in place. mask is bool.
T_MASKED_FILL_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = at::infer_size(self.sizes(), mask.sizes());
  // avoid clone()/empty_like (not registered for ascend): alloc + copy_.
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  out.copy_(self.expand(out_shape));
  auto mask_b = mask.expand(out_shape).contiguous();

  ascend::AclTensorWrapper acl_self(out);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());

  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_mask.get(),
      acl_value.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# masked_fill.Tensor: (self, mask, value) -> Tensor. value is a 0-dim tensor;
#   coerce to self's device/dtype (CPU scalar-tensor path, cf. binary prologue).
T_MASKED_FILL_TENSOR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = at::infer_size(self.sizes(), mask.sizes());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  out.copy_(self.expand(out_shape));
  auto mask_b = mask.expand(out_shape).contiguous();
  auto value_c = value.is_privateuseone()
      ? (value.scalar_type() == self.scalar_type() ? value : value.to(self.scalar_type()))
      : value.to(self.options());

  ascend::AclTensorWrapper acl_self(out);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclTensorWrapper acl_value(value_c);

  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_mask.get(),
      acl_value.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gather: (self, dim, index, sparse_grad) -> Tensor (= index shape, self dtype).
#   aclnnGather(self, dim, index, out). dim normalized for negatives.
T_GATHER = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, bool sparse_grad) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      index.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_index(index);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_index.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# index_select: (self, dim, index) -> Tensor (= self shape w/ dim replaced by
#   index.numel()). index is 1-D. aclnnIndexSelect(self, dim, index, out).
T_INDEX_SELECT = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
  out_shape[d] = index.numel();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_index(index);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_index.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# zero_: (self) -> self&, in-place. aclnnInplaceZero(selfRef). selfRef is both
#   input and output; return the same tensor. Factory ops (zeros/*_like) build
#   the storage then call this, so it must be device-side aclnn (no h2d).
T_INPLACE_ZERO = """\
at::Tensor& {kernel}(at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()));
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# fill_.Scalar: (self, value) -> self&, in-place. aclnnInplaceFillScalar(
#   selfRef, value). Scalar coerced to self's dtype.
T_INPLACE_FILL_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# fill_.Tensor: (self, value) -> self&, in-place. aclnnInplaceFillTensor(
#   selfRef, value). value is a 0-dim tensor; coerce to self device/dtype
#   (CPU scalar-tensor path, cf. masked_fill.Tensor prologue).
T_INPLACE_FILL_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto value_c = value.is_privateuseone()
      ? (value.scalar_type() == self.scalar_type() ? value : value.to(self.scalar_type()))
      : value.to(self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_value(value_c);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# embedding: (weight, indices, padding_idx, scale_grad_by_freq, sparse) -> Tensor.
#   aclnnEmbedding(weight, indices, out) uses only weight+indices; the trailing
#   three args are ignored by aclnn. Output = indices.sizes() + [weight.size(1)].
T_EMBEDDING = """\
at::Tensor {kernel}(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_sizes = indices.sizes().vec();
  out_sizes.push_back(weight.size(1));
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, weight.options());

  ascend::AclTensorWrapper acl_weight(weight);
  ascend::AclTensorWrapper acl_indices(indices);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_weight.get(), acl_indices.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# embedding_dense_backward: (grad, indices, num_weights, padding_idx,
#   scale_grad_by_freq) -> grad_weight = {num_weights, grad.size(-1)}.
#   aclnnEmbeddingDenseBackward(grad, indices, numWeights, paddingIdx,
#   scaleGradByFreq, out); numWeights/paddingIdx passed as int64 (aclnn uint64).
T_EMBEDDING_DENSE_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      {{num_weights, grad_output.size(-1)}}, grad_output.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_indices(indices);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_indices.get(),
      num_weights, padding_idx, scale_grad_by_freq, acl_grad_weight.get());
  return grad_weight;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# constant_pad_nd: (self, pad, value) -> Tensor. aclnnConstantPadNd(self, pad,
#   value, out). Output widens the trailing dims by pad pairs (last-dim-first,
#   matching torch's pad ordering).
T_CONSTANT_PAD_ND = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto input_sizes = self.sizes().vec();
  auto ndim = input_sizes.size();
  auto pad_size = pad.size();
  std::vector<int64_t> out_sizes(input_sizes.begin(), input_sizes.end());
  for (size_t i = 0; i < pad_size / 2; ++i) {{
    auto dim = ndim - 1 - i;
    out_sizes[dim] += pad[2 * i] + pad[2 * i + 1];
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclIntArrayWrapper acl_pad(pad);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_pad.get(), acl_value.get(),
      acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# avg_pool2d_backward: (grad_output, self, k, stride, padding, ceil_mode,
#   count_include_pad, divisor_override) -> grad_input (= self shape). Same
#   NCHW-format requirement as the forward. cubeMathType=0 (KEEP_DTYPE).
T_AVG_POOL2D_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  int64_t div_override = divisor_override.value_or(0);
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_k.get(), acl_s.get(),
      acl_p.get(), ceil_mode, count_include_pad, div_override, (int8_t)0,
      acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _adaptive_avg_pool2d_backward: (grad_output, self) -> grad_input (= self
#   shape). Minimal aclnn(grad, self, out). NCHW format like the forward.
T_ADAPTIVE_AVG_POOL2D_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_layer_norm_backward: (grad_out, input, normalized_shape, mean, rstd,
#   weight?, bias?, output_mask[3]) -> (grad_input, grad_weight, grad_bias).
#   grad_input = input shape; grad_weight/grad_bias = normalized_shape.
T_NATIVE_LAYER_NORM_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_out, const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& mean, const at::Tensor& rstd, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      normalized_shape, input.options());
  auto grad_bias = ascend::OpPreparation::apply_tensor_without_format(
      normalized_shape, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_grad(grad_out);
  ascend::AclTensorWrapper acl_input(input);
  ascend::AclIntArrayWrapper acl_nshape(normalized_shape);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);
  ascend::AclTensorWrapper acl_grad_bias(grad_bias);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_nshape.get(),
      acl_mean.get(), acl_rstd.get(), acl_weight.get(), acl_bias.get(),
      acl_mask.get(), acl_grad_input.get(), acl_grad_weight.get(),
      acl_grad_bias.get());
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_group_norm_backward: (grad_out, input, mean, rstd, weight?, N, C, HxW,
#   group, output_mask[3]) -> (grad_input, grad_gamma, grad_beta). grad_input =
#   input shape; grad_gamma/grad_beta = [C].
T_NATIVE_GROUP_NORM_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean, const at::Tensor& rstd, const ::std::optional<at::Tensor>& weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_gamma = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());
  auto grad_beta = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_grad(grad_out);
  ascend::AclTensorWrapper acl_input(input);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);
  ascend::AclTensorWrapper acl_gamma(weight_t);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input);
  ascend::AclTensorWrapper acl_grad_gamma(grad_gamma);
  ascend::AclTensorWrapper acl_grad_beta(grad_beta);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_mean.get(),
      acl_rstd.get(), acl_gamma.get(), N, C, HxW, group, acl_mask.get(),
      acl_grad_input.get(), acl_grad_gamma.get(), acl_grad_beta.get());
  return std::make_tuple(grad_input, grad_gamma, grad_beta);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# max_pool2d_with_indices_backward: (grad_output, self, kernel_size, stride,
#   padding, dilation, ceil_mode, indices) -> grad_input (= self shape).
#   NOTE aten arg order != aclnn: aclnn puts `indices` right after `self`
#   (aclnn<Name>(grad, self, indices, k, s, p, dil, ceil, gradInput)).
T_MAX_POOL2D_INDICES_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor& indices) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());
  std::vector<int64_t> dil(dilation.begin(), dilation.end());

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  // aclnn max_pool2d backward (unlike forward) rejects ND 4-D; tag NCHW.
  // aclnn also wants indices as int32, but the forward emits int64 -> cast.
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  auto indices_i32 = indices.scalar_type() == at::kInt ? indices : indices.to(at::kInt);
  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclTensorWrapper acl_indices(indices_i32, fmt);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclIntArrayWrapper acl_dil(dil);
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_indices.get(),
      acl_k.get(), acl_s.get(), acl_p.get(), acl_dil.get(), ceil_mode,
      acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_batch_norm: (input, weight?, bias?, running_mean?, running_var?,
#   training, momentum, eps) -> (output, save_mean, save_invstd). runningMean/
#   runningVar are updated in-place by aclnn (non-const). save_mean/save_invstd
#   are [C]. Needs NCHW/NCL/NCDHW format like conv. momentum/eps are by-value
#   double (varargs-safe: aarch64 already passes double, cf. group_norm).
T_NATIVE_BATCH_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& input, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, const ::std::optional<at::Tensor>& running_mean, const ::std::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t rank = input.dim();
  int64_t C = input.size(1);
  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : (rank == 5 ? ACL_FORMAT_NCDHW : ACL_FORMAT_ND));

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto save_mean = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());
  auto save_invstd = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());
  at::Tensor rmean_t = running_mean.value_or(at::Tensor());
  at::Tensor rvar_t = running_var.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclTensorWrapper acl_rmean(rmean_t);
  ascend::AclTensorWrapper acl_rvar(rvar_t);
  ascend::AclTensorWrapper acl_out(out, fmt);
  ascend::AclTensorWrapper acl_save_mean(save_mean);
  ascend::AclTensorWrapper acl_save_invstd(save_invstd);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_weight.get(), acl_bias.get(),
      const_cast<aclTensor*>(acl_rmean.get()), const_cast<aclTensor*>(acl_rvar.get()),
      training, momentum, eps, acl_out.get(), acl_save_mean.get(), acl_save_invstd.get());
  return std::make_tuple(out, save_mean, save_invstd);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_batch_norm_backward: (grad_out, input, weight?, running_mean?,
#   running_var?, save_mean?, save_invstd?, train, eps, output_mask[3]) ->
#   (grad_input, grad_weight, grad_bias). grad_weight/grad_bias are [C].
T_NATIVE_BATCH_NORM_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_out, const at::Tensor& input, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& running_mean, const ::std::optional<at::Tensor>& running_var, const ::std::optional<at::Tensor>& save_mean, const ::std::optional<at::Tensor>& save_invstd, bool train, double eps, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t rank = input.dim();
  int64_t C = input.size(1);
  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : (rank == 5 ? ACL_FORMAT_NCDHW : ACL_FORMAT_ND));

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());
  auto grad_bias = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor rmean_t = running_mean.value_or(at::Tensor());
  at::Tensor rvar_t = running_var.value_or(at::Tensor());
  at::Tensor smean_t = save_mean.value_or(at::Tensor());
  at::Tensor sinvstd_t = save_invstd.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_grad(grad_out, fmt);
  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_rmean(rmean_t);
  ascend::AclTensorWrapper acl_rvar(rvar_t);
  ascend::AclTensorWrapper acl_smean(smean_t);
  ascend::AclTensorWrapper acl_sinvstd(sinvstd_t);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);
  ascend::AclTensorWrapper acl_grad_bias(grad_bias);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_weight.get(),
      acl_rmean.get(), acl_rvar.get(), acl_smean.get(), acl_sinvstd.get(),
      train, eps, acl_mask.get(), acl_grad_input.get(), acl_grad_weight.get(),
      acl_grad_bias.get());
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# convolution: (input, weight, bias?, stride, padding, dilation, transposed,
#   output_padding, groups) -> Tensor. Non-transposed conv only (transposed
#   uses output_padding + a different out formula -> left to a later batch).
#   out[N, Cout, *spatial] where Cout=weight.size(0). aclnnConvolution wants
#   NCHW/NCL/NCDHW format (ND 4-D is rejected, same as pooling). cubeMathType=0
#   (KEEP_DTYPE) keeps full fp32 -- type=1 (ALLOW_FP32_DOWN_PRECISION) loses
#   ~2.5e-3 vs CPU on the cube unit.
T_CONVOLUTION = """\
at::Tensor {kernel}(const at::Tensor& input, const at::Tensor& weight, const ::std::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(!transposed, "convolution codegen kernel: transposed not supported");
  int64_t rank = input.dim();
  int64_t nspatial = rank - 2;
  auto out_shape = std::vector<int64_t>{{input.size(0), weight.size(0)}};
  for (int64_t i = 0; i < nspatial; ++i) {{
    int64_t in = input.size(i + 2), k = weight.size(i + 2);
    int64_t st = stride.size() == 1 ? stride[0] : stride[i];
    int64_t pd = padding.size() == 1 ? padding[0] : padding[i];
    int64_t dl = dilation.size() == 1 ? dilation[0] : dilation[i];
    out_shape.push_back((in + 2 * pd - dl * (k - 1) - 1) / st + 1);
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, input.options());

  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : ACL_FORMAT_NCDHW);
  at::Tensor bias_t = bias.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight, fmt);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclIntArrayWrapper acl_stride(stride);
  ascend::AclIntArrayWrapper acl_padding(padding);
  ascend::AclIntArrayWrapper acl_dilation(dilation);
  ascend::AclIntArrayWrapper acl_outpad(output_padding);
  ascend::AclTensorWrapper acl_out(out, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_weight.get(), acl_bias.get(),
      acl_stride.get(), acl_padding.get(), acl_dilation.get(), transposed,
      acl_outpad.get(), groups, acl_out.get(), (int8_t)0);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# convolution_backward: (grad_output, input, weight, bias_sizes?, stride,
#   padding, dilation, transposed, output_padding, groups, output_mask[3]) ->
#   (grad_input, grad_weight, grad_bias). aclnn allocates all three; unwanted
#   ones (output_mask=false) are still passed but ignored. grad_bias has shape
#   [Cout] = weight.size(0).
T_CONVOLUTION_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(!transposed, "convolution_backward codegen kernel: transposed not supported");
  int64_t rank = input.dim();
  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : ACL_FORMAT_NCDHW);

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      weight.sizes(), weight.options());
  std::vector<int64_t> bias_shape = bias_sizes.has_value()
      ? bias_sizes.value().vec() : std::vector<int64_t>{{weight.size(0)}};
  auto grad_bias = ascend::OpPreparation::apply_tensor_without_format(
      bias_shape, weight.options());

  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight, fmt);
  ascend::AclIntArrayWrapper acl_bias_sizes(bias_shape);
  ascend::AclIntArrayWrapper acl_stride(stride);
  ascend::AclIntArrayWrapper acl_padding(padding);
  ascend::AclIntArrayWrapper acl_dilation(dilation);
  ascend::AclIntArrayWrapper acl_outpad(output_padding);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight, fmt);
  ascend::AclTensorWrapper acl_grad_bias(grad_bias);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_weight.get(),
      acl_bias_sizes.get(), acl_stride.get(), acl_padding.get(),
      acl_dilation.get(), transposed, acl_outpad.get(), (int)groups,
      acl_mask.get(), (int8_t)0, acl_grad_input.get(), acl_grad_weight.get(),
      acl_grad_bias.get());
  return std::make_tuple(grad_input, grad_weight, grad_bias);
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
    "gemm_addmv":          T_GEMM_ADDMV,
    "gemm_addr":           T_GEMM_ADDR,
    "mv":                  T_MV,
    "dot":                 T_DOT,
    "bce":                 T_BCE,
    "bce_backward":        T_BCE_BACKWARD,
    "bce_logits":          T_BCE_LOGITS,
    "layer_norm":          T_LAYER_NORM,
    "group_norm":          T_GROUP_NORM,
    "gelu":                T_GELU,
    "gelu_backward":       T_GELU_BACKWARD,
    "log_softmax":         T_LOG_SOFTMAX,
    "softmax_backward":    T_SOFTMAX_BACKWARD,
    "binary_scalar":       T_BINARY_SCALAR,
    "act_backward_self":   T_ACT_BACKWARD_SELF,
    "where":               T_WHERE,
    "softmax_fwd":         T_SOFTMAX_FWD,
    "reduce_all":          T_REDUCE_ALL,
    "reduce_sum_dtype":    T_REDUCE_SUM_DTYPE,
    "reduce_mean_dtype":   T_REDUCE_MEAN_DTYPE,
    "adaptive_avg_pool2d": T_ADAPTIVE_AVG_POOL2D,
    "avg_pool2d":          T_AVG_POOL2D,
    "max_pool2d_indices":  T_MAX_POOL2D_INDICES,
    "convolution":         T_CONVOLUTION,
    "convolution_backward": T_CONVOLUTION_BACKWARD,
    "max_pool2d_indices_backward": T_MAX_POOL2D_INDICES_BACKWARD,
    "native_batch_norm":   T_NATIVE_BATCH_NORM,
    "native_batch_norm_backward": T_NATIVE_BATCH_NORM_BACKWARD,
    "avg_pool2d_backward": T_AVG_POOL2D_BACKWARD,
    "adaptive_avg_pool2d_backward": T_ADAPTIVE_AVG_POOL2D_BACKWARD,
    "native_layer_norm_backward": T_NATIVE_LAYER_NORM_BACKWARD,
    "native_group_norm_backward": T_NATIVE_GROUP_NORM_BACKWARD,
    "masked_fill_scalar":  T_MASKED_FILL_SCALAR,
    "masked_fill_tensor":  T_MASKED_FILL_TENSOR,
    "gather":              T_GATHER,
    "index_select":        T_INDEX_SELECT,
    "inplace_zero":        T_INPLACE_ZERO,
    "inplace_fill_scalar": T_INPLACE_FILL_SCALAR,
    "inplace_fill_tensor": T_INPLACE_FILL_TENSOR,
    "embedding":           T_EMBEDDING,
    "embedding_dense_backward": T_EMBEDDING_DENSE_BACKWARD,
    "constant_pad_nd":     T_CONSTANT_PAD_ND,
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
