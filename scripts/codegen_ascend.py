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
