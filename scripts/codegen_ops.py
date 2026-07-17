#!/usr/bin/env python3
"""
Codegen for torch_fl CUDA operators.

Reads:
  - torch_fl/backends_cuda.conf (op list + backend mapping)
  - PyTorch native_functions.yaml (via torchgen)

Generates (into csrc/aten/generated/):
  - ops.h            shared dispatcher declarations (typedef + DECLARE_DISPATCHER)
  - ops.cc           dispatcher definitions (ADD_IMPL_TO_DISPATCHER)
  - cuda_kernels.cc  CUDA boxing kernels + REGISTER_IMPL_TO_DISPATCHER
  - register.inc     wrapper functions + m.impl() lines for register.cc

Authoritative signature source: torchgen's faithful C++ signature
(exploded TensorOptions form, required for PrivateUse1 boxing dispatch).
We never hand-roll a type table; torchgen produces the exact `at::` signature.

Naming convention (schema-driven, PyTorch-aligned):
  "add.Tensor"           -> AddTensorFn                 / add_tensor_dispatcher
  "add_.Tensor"          -> AddInplaceTensorFn          / add_inplace_tensor_dispatcher
  "mm.out"               -> MmOutFn                     / mm_out_dispatcher
  "_foreach_add_.Scalar" -> ForeachAddInplaceScalarFn   / foreach_add_inplace_scalar_dispatcher

Categories (detected from torchgen metadata):
  functional_pure    standard DeviceBoxingGuard + at::op call
  inplace            box + at::op_ inplace + unbox (returns Tensor& or void)
  out_variant        box(self.., out) + at::op_out(out, ...) + return out
  tuple_return       box + call + unbox each tuple element
  foreach_tensorlist TensorListBoxingGuard + call (+ unbox vec for non-inplace)
  factory            create device tensor directly (no input boxing)
  special_optlist    box self + each optional<Tensor> in the list + call + unbox
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Operators requiring ArrayRef instead of IListRef (empirically determined from PyTorch 2.13 dispatcher)
# These operators have CompositeExplicitAutograd kernels registered with ArrayRef signatures.
# General pattern: ALL _foreach_* ops use ArrayRef (CompositeExplicitAutograd dispatch key)
# Only aten::cat uses IListRef (Batched dispatch key)
ARRAYREF_OPS = {
    "_foreach_add_.List",
    "_foreach_add_.Scalar",
    "_foreach_add_.ScalarList",
    "_foreach_sub_.List",
    "_foreach_mul_.List",
    "_foreach_mul_.Scalar",
    "_foreach_mul_.ScalarList",
    "_foreach_div_.List",
    "_foreach_div_.ScalarList",
    "_foreach_abs_",
    "_foreach_neg_",
    "_foreach_neg",
    "_foreach_sqrt_",
    "_foreach_sqrt",
    "_foreach_reciprocal_",
    "_foreach_reciprocal",
    "_foreach_zero_",
    "_foreach_add.List",
    "_foreach_mul.List",
    "_foreach_addcdiv_.ScalarList",
    "_foreach_addcmul_.Scalar",
    "_foreach_lerp_.Scalar",
}

def should_use_arrayref(func_name):
    """Check if operator needs ArrayRef instead of IListRef to match PyTorch 2.13 dispatcher."""
    return func_name in ARRAYREF_OPS

try:
    import torchgen
    from torchgen.gen import parse_native_yaml
    from torchgen.api import cpp
    from torchgen.api.types import CppSignatureGroup
    from torchgen import local
except ImportError:
    print("Error: torchgen not found. Install torch>=2.0 and pyyaml.", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Naming conventions (schema-driven, PyTorch-aligned)
# ============================================================================

def schema_to_cpp_name(op_name: str) -> Tuple[str, str]:
    """
    (fn_type PascalCase, dispatcher_name snake_case) from a schema op name.

      "add.Tensor"           -> ("AddTensorFn", "add_tensor_dispatcher")
      "add_.Tensor"          -> ("AddInplaceTensorFn", "add_inplace_tensor_dispatcher")
      "mm.out"               -> ("MmOutFn", "mm_out_dispatcher")
      "_foreach_add_.Scalar" -> ("ForeachAddInplaceScalarFn", "foreach_add_inplace_scalar_dispatcher")
    """
    parts = op_name.split('.')
    base = parts[0]
    variant = parts[1] if len(parts) > 1 else None

    is_foreach = base.startswith('_foreach_')
    base_clean = base.lstrip('_')
    is_inplace = base_clean.endswith('_')
    if is_inplace:
        base_clean = base_clean.rstrip('_')

    # --- PascalCase type name ---
    if is_foreach:
        core = base_clean[len('foreach_'):]
        type_base = 'Foreach' + ''.join(w.capitalize() for w in core.split('_') if w)
    else:
        type_base = ''.join(w.capitalize() for w in base_clean.split('_') if w)
    if is_inplace:
        type_base += 'Inplace'
    if variant:
        type_base += ''.join(w.capitalize() for w in variant.split('_') if w)
    fn_type = type_base + 'Fn'

    # --- snake_case dispatcher name ---
    disp = base_clean  # foreach already normalized (leading _ stripped)
    if is_inplace:
        disp += '_inplace'
    if variant:
        disp += '_' + variant.lower()
    dispatcher_name = disp + '_dispatcher'

    return fn_type, dispatcher_name


def kernel_name(fn_type: str) -> str:
    """AddTensorFn -> AddTensorKernelCuda"""
    return fn_type[:-2] + 'KernelCuda'


# ============================================================================
# Category detection (torchgen metadata)
# ============================================================================

def detect_category(func) -> str:
    s = func.func
    arg_types = [str(a.type) for a in s.arguments.flat_all]

    has_tensorlist = any(t == "Tensor[]" for t in arg_types)
    has_optlist = any("Tensor?[]" in t for t in arg_types)
    has_tensor_in = any(t.startswith("Tensor") and "[]" not in t for t in arg_types)

    # Special case: new_* ops are factory-like (use at::empty + fill)
    op_name = str(s.name.name)
    if op_name.startswith("new_"):
        return "factory"

    if has_optlist:
        return "special_optlist"
    if not has_tensor_in and not has_tensorlist:
        return "factory"
    if has_tensorlist:
        return "foreach_tensorlist"
    if s.is_out_fn():
        return "out_variant"
    if str(s.kind()).split('.')[-1] == "inplace":
        return "inplace"
    if len(s.returns) > 1:
        return "tuple_return"
    return "functional_pure"


# ============================================================================
# Authoritative signatures via torchgen (faithful = exploded TensorOptions)
# ============================================================================

def unified_sig(func):
    """
    CppSignature (faithful) for ALL uses: typedef, kernel, and wrapper.
    This ensures the typedef, REGISTER_IMPL, and m.impl all agree on types.

    Faithful = TensorOptions exploded into dtype/layout/device/pin_memory,
    which is exactly what PrivateUse1 boxing dispatch requires.

    Returns (ptr_type_str, ret_type_str, [(cpp_type_str, name), ...]).
    Must be called inside a `with local.parametrize(...)` context.
    """
    from torchgen.api.types import CppSignatureGroup
    group = CppSignatureGroup.from_native_function(func, method=False, fallback_binding=False)
    sig = group.faithful_signature if group.faithful_signature is not None else group.signature
    ptr = sig.ptr_type()
    ret_type = ptr.split('(*)', 1)[0].strip()
    args = [(a.type, a.name) for a in sig.arguments()]
    return ptr, ret_type, args


def fn_ptr_signature(ret_type: str, args: List[Tuple[str, str]]) -> str:
    return f"{ret_type} (*)({', '.join(t for t, _ in args)})"


def args_decl(args: List[Tuple[str, str]]) -> str:
    return ", ".join(f"{t} {n}" for t, n in args)


def call_args(args: List[Tuple[str, str]]) -> str:
    return ", ".join(n for _, n in args)


def at_api_base(op_name: str) -> str:
    """schema op -> at:: function base name. 'mm.out'->'mm', '_foreach_add_.Scalar'->'_foreach_add_'."""
    return op_name.split('.')[0]


# ============================================================================
# Per-category kernel templates
# ============================================================================

def tensor_arg_names(args: List[Tuple[str, str]]) -> List[str]:
    """Plain (non-optional, non-list) at::Tensor args - safe for DeviceBoxingGuard."""
    out = []
    for t, n in args:
        if "Tensor" not in t:
            continue
        if "optional" in t or "List" in t or "TensorList" in t or "ArrayRef<at::Tensor" in t:
            continue
        out.append(n)
    return out


def optional_tensor_names(args: List[Tuple[str, str]]) -> List[str]:
    return [n for t, n in args if "optional<at::Tensor>" in t]


def gen_functional_pure(op, fn_type, ret_type, args):
    kn = kernel_name(fn_type)
    guard = ", ".join(tensor_arg_names(args))
    api = f"at::{at_api_base(op)}"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  auto result = {api}({call_args(args)});
  UnboxToFlagos(result);
  return result;
}}"""


def gen_inplace(op, fn_type, ret_type, args):
    """add_.Tensor / fill_.Scalar / masked_fill_.Scalar: mutate first tensor, return it (or void)."""
    kn = kernel_name(fn_type)
    tensors = tensor_arg_names(args)
    guard = ", ".join(tensors)
    # Inplace ops use method syntax: self.add_(other, alpha), NOT at::add_(...)
    base = at_api_base(op)  # e.g. "add_" already has underscore for add_.Tensor
    method = base  # keep trailing underscore
    self_name = args[0][1]
    other_args = ", ".join(n for _, n in args[1:])
    body_call = f"{self_name}.{method}({other_args});" if other_args else f"{self_name}.{method}();"
    if ret_type == "void":
        ret_line = ""
    else:
        ret_line = f"\n  return {self_name};"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  {body_call}{ret_line}
}}"""


def gen_out_variant(op, fn_type, ret_type, args):
    """mm.out / bmm.out: faithful arg order is (self, mat2, out); call at::op_out(out, self, mat2)."""
    kn = kernel_name(fn_type)
    guard = ", ".join(tensor_arg_names(args))
    base = at_api_base(op)
    api = f"at::{base}_out"
    # out is the mutable Tensor& arg; find it, put it first
    out_name = None
    other = []
    for t, n in args:
        if "Tensor &" in t and "const" not in t:
            out_name = n
        else:
            other.append(n)
    ordered = ", ".join([out_name] + other)
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  {api}({ordered});
  return {out_name};
}}"""


def gen_tuple_return(op, fn_type, ret_type, args):
    """nll_loss_forward / sort / topk: unbox each tuple element."""
    kn = kernel_name(fn_type)
    # optional<Tensor> weight needs a holder to be boxed by DeviceBoxingGuard
    opt_names = optional_tensor_names(args)
    plain = tensor_arg_names(args)
    api = f"at::{at_api_base(op)}"
    ntuple = ret_type.count(',') + 1  # ::std::tuple<a,b> -> 2

    holder_lines = ""
    guard_names = list(plain)
    for on in opt_names:
        holder_lines += f"  at::Tensor {on}_t = {on}.has_value() ? *{on} : at::Tensor();\n"
        guard_names.append(f"{on}_t")
    guard = ", ".join(guard_names)

    unbox_lines = "\n".join(f"  UnboxToFlagos(std::get<{i}>(result));" for i in range(ntuple))
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
  auto result = {api}({call_args(args)});
{unbox_lines}
  return result;
}}"""


def gen_foreach(op, fn_type, ret_type, args):
    """cat + _foreach_*: materialize ITensorListRef, box, call API, unbox result."""
    kn = kernel_name(fn_type)
    api = f"at::{at_api_base(op)}"

    # Detect TensorList args (ITensorListRef in torch 2.13)
    tensorlist_args = [(t, n) for t, n in args if "TensorList" in t]

    # Materialize ITensorListRef → std::vector<Tensor>
    # The vector implicitly converts to TensorList (ArrayRef) for PyTorch API
    materialize_lines = ""
    call_arg_names = []
    for t, n in args:
        if "ITensorListRef" in t or "TensorList" in t:
            mat_name = f"{n}_vec"
            materialize_lines += f"  auto {mat_name} = MaterializeToTensorVec({n});\n"
            call_arg_names.append(mat_name)
        else:
            call_arg_names.append(n)

    call_args_str = ", ".join(call_arg_names)

    # Box the materialized vectors
    box_lines = ""
    for t, n in tensorlist_args:
        mat_name = f"{n}_vec"
        box_lines += f"  guard.box({mat_name});\n"

    if ret_type == "void":
        body = f"  {api}({call_args_str});"
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}{body}
}}"""
    elif "vector" in ret_type:
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}  auto result = {api}({call_args_str});
  UnboxTensorVecToFlagos(result);
  return result;
}}"""
    else:
        # single Tensor return (cat)
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}  auto result = {api}({call_args_str});
  UnboxToFlagos(result);
  return result;
}}"""


def gen_factory(op, fn_type, ret_type, args):
    """
    zeros / scalar_tensor / arange / arange.start_step / new_ones: build device tensor directly.
    Faithful args end with dtype/layout/device/pin_memory. Create via at::empty on
    the requested (or PrivateUse1) device, then fill.
    """
    kn = kernel_name(fn_type)
    names = [n for _, n in args]
    has_self = args and "at::Tensor" in args[0][0]

    # option field names present in the faithful signature
    def opt(name, default):
        return f"{name}.value_or({default})" if name in names else default

    dtype_default = f"{names[0]}.scalar_type()" if has_self else "at::kFloat"
    layout_default = f"{names[0]}.layout()" if has_self else "at::kStrided"
    device_default = f"{names[0]}.device()" if has_self else "at::Device(at::kPrivateUse1, 0)"

    options = (
        "  auto options = at::TensorOptions()\n"
        f"    .dtype({opt('dtype', dtype_default)})\n"
        f"    .layout({opt('layout', layout_default)})\n"
        f"    .device({opt('device', device_default)})\n"
        f"    .pinned_memory({opt('pin_memory', 'false')});"
    )

    base = at_api_base(op)
    if base == "zeros":
        make = f"  auto result = at::empty({names[0]}, options);\n  result.zero_();"
    elif base == "scalar_tensor":
        make = f"  auto result = at::empty({{}}, options);\n  result.fill_({names[0]});"
    elif base == "new_ones":
        # new_ones is not in public at:: API; use at::empty + fill_ like hand-written code
        size_arg = names[1]  # (self, size, dtype, layout, device, pin_memory)
        make = f"  auto result = at::empty({size_arg}, options);\n  result.fill_(1);"
    elif base == "arange":
        # arange computes a sequence, so we must call the real at::arange kernel.
        # If we call it with a PrivateUse1 device, it dispatches back to THIS kernel
        # -> infinite recursion -> stack overflow. Instead, build on CUDA (hits the
        # external libtorch_cuda.so kernel), then unbox the result back to flagos.
        scalar_args = [n for t, n in args if t == "const at::Scalar &"]
        cuda_options = (
            "  auto cuda_options = options.device(\n"
            "      options.device().type() == at::kPrivateUse1\n"
            "          ? at::Device(at::kCUDA, options.device().index())\n"
            "          : options.device());"
        )
        make = (
            f"{cuda_options}\n"
            f"  auto result = at::arange({', '.join(scalar_args)}, cuda_options);\n"
            f"  if (result.device().type() == at::kCUDA) UnboxToFlagos(result);"
        )
    else:
        make = f"  auto result = at::empty({names[0] if has_self else '{}'}, options);"

    return f"""{ret_type} {kn}({args_decl(args)}) {{
{options}
{make}
  return result;
}}"""


def gen_optlist(op, fn_type, ret_type, args):
    """index.Tensor: box self + each defined optional<Tensor> in the list."""
    kn = kernel_name(fn_type)
    self_name = args[0][1]
    list_name = args[1][1]
    api = f"at::{at_api_base(op)}"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  BoxToCuda({self_name});
  std::vector<at::Tensor> boxed_holders;
  for (int64_t i = 0; i < static_cast<int64_t>({list_name}.size()); ++i) {{
    auto opt = {list_name}.get(i);
    if (opt.has_value() && opt->defined()) {{
      BoxToCuda(*opt);
      boxed_holders.push_back(*opt);
    }}
  }}
  auto result = {api}({self_name}, {list_name});
  UnboxToFlagos({self_name});
  for (auto& t : boxed_holders) {{
    UnboxToFlagos(t);
  }}
  UnboxToFlagos(result);
  return result;
}}"""


CATEGORY_GENERATORS = {
    "functional_pure": gen_functional_pure,
    "inplace": gen_inplace,
    "out_variant": gen_out_variant,
    "tuple_return": gen_tuple_return,
    "foreach_tensorlist": gen_foreach,
    "factory": gen_factory,
    "special_optlist": gen_optlist,
}


# ============================================================================
# Wrapper functions (register.cc side)
# ============================================================================

def gen_wrapper(op, fn_type, dispatcher, ret_type, args):
    """Wrapper<Name> that forwards into the dispatcher (used by TORCH_LIBRARY_IMPL)."""
    wname = "Wrapper" + fn_type[:-2]
    ns = "at::native::flagos::"
    call = f"{ns}{dispatcher}({call_args(args)})"
    if ret_type == "void":
        body = f"  {call};"
    else:
        body = f"  return {call};"
    return f"{ret_type} {wname}({args_decl(args)}) {{\n{body}\n}}", wname


# ============================================================================
# at:: header includes needed by cuda_kernels.cc
# ============================================================================

def api_headers(ops: List[str]) -> List[str]:
    bases = set()
    for op in ops:
        base = at_api_base(op).rstrip('_')  # add_ -> add ; _foreach_add_ -> _foreach_add
        # keep leading underscore form for foreach/softmax etc.
        raw = at_api_base(op).rstrip('_')
        bases.add(raw)
        # out variants need the _out header too (same file)
    hdrs = []
    for b in sorted(bases):
        hdrs.append(f"#include <ATen/ops/{b}.h>")
    return hdrs


# ============================================================================
# Main
# ============================================================================

def main():
    repo_root = Path(__file__).parent.parent
    conf_path = repo_root / "torch_fl/backends_cuda.conf"
    out_dir = repo_root / "csrc/aten/generated"
    out_dir.mkdir(exist_ok=True)

    print("Loading configuration and schemas...")
    ops = []
    for line in conf_path.read_text().splitlines():
        line = line.split('#')[0].strip()
        if not line or '=' not in line:
            continue
        op, backend = line.split('=', 1)
        if backend.strip() == "cuda":
            ops.append(op.strip())

    root = Path(torchgen.__file__).parent
    nf = parse_native_yaml(
        str(root / "packaged/ATen/native/native_functions.yaml"),
        str(root / "packaged/ATen/native/tags.yaml"),
    )
    funcs = {str(f.func.name): f for f in nf.native_functions}

    print(f"Found {len(ops)} ops in backends_cuda.conf")

    op_info = {}
    categories = defaultdict(list)

    # Generate signatures per-operator with correct IListRef/ArrayRef setting
    for op in ops:
        if op not in funcs:
            print(f"  WARNING: {op} not in native_functions.yaml", file=sys.stderr)
            continue
        func = funcs[op]
        cat = detect_category(func)
        fn_type, dispatcher = schema_to_cpp_name(op)

        # Determine if this op needs ArrayRef (vs IListRef default)
        use_arrayref = should_use_arrayref(op)

        # Generate signature with operator-specific IListRef/ArrayRef setting
        with local.parametrize(
            use_const_ref_for_mutable_tensors=False,
            use_ilistref_for_tensor_lists=not use_arrayref,  # False=ArrayRef, True=IListRef
        ):
            # Use unified CppSignature (faithful) for typedef, kernel, AND wrapper
            ptr_type, ret_type, args = unified_sig(func)

            categories[cat].append(op)
            op_info[op] = dict(
                fn_type=fn_type, dispatcher=dispatcher, category=cat,
                ptr_type=ptr_type, ret_type=ret_type, args=args,
                func=func,
            )

    print("\nCategory breakdown:")
    for cat in sorted(categories):
        print(f"  {cat:20s} {len(categories[cat]):3d} ops")

    # ---- ops.h ----
    print("\nGenerating ops.h...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        "#pragma once",
        "",
        "#include <ATen/core/Tensor.h>",
        "#include \"../dispatcher.h\"",
        "",
        "namespace at::native::flagos {",
        "",
    ]
    for op in sorted(op_info):
        i = op_info[op]
        lines.append(f"using {i['fn_type']} = {i['ptr_type']};")
        lines.append(f"DECLARE_DISPATCHER({i['fn_type']}, {i['dispatcher']})")
        lines.append("")
    lines.append("} // namespace at::native::flagos")
    (out_dir / "ops.h").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} declarations")

    # ---- ops.cc ----
    print("Generating ops.cc...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        "#include \"ops.h\"",
        "",
        "namespace at::native::flagos {",
        "",
    ]
    for op in sorted(op_info):
        i = op_info[op]
        lines.append(f'ADD_IMPL_TO_DISPATCHER({i["fn_type"]}, {i["dispatcher"]}, "{op}")')
    lines.append("")
    lines.append("} // namespace at::native::flagos")
    (out_dir / "ops.cc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} definitions")

    # ---- cuda_kernels.cc ----
    print("Generating cuda_kernels.cc...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        "#include \"ops.h\"",
        "#include \"../device_boxing.h\"",
        "",
        "#include <vector>",
        "#include <tuple>",
        "",
    ]
    lines += api_headers(list(op_info.keys()))
    lines += [
        "",
        "namespace at::native::flagos {",
        "namespace {",
        "",
    ]
    for op in sorted(op_info):
        i = op_info[op]
        gen = CATEGORY_GENERATORS[i["category"]]
        lines.append(gen(op, i["fn_type"], i["ret_type"], i["args"]))
        lines.append("")
    lines.append("} // namespace")
    lines.append("")
    for op in sorted(op_info):
        i = op_info[op]
        kn = kernel_name(i["fn_type"])
        lines.append(f'REGISTER_IMPL_TO_DISPATCHER({i["fn_type"]}, {i["dispatcher"]}, Backend::kCuda, {kn})')
    lines.append("")
    lines.append("} // namespace at::native::flagos")
    (out_dir / "cuda_kernels.cc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} kernels")

    # ---- register.inc ----
    print("Generating register.inc...")
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "// Included by register.cc: wrapper fns + m.impl() lines.",
        "",
        "// ---- wrapper functions ----",
        "#ifdef FLAGOS_GEN_WRAPPERS",
    ]
    impl_lines = []
    for op in sorted(op_info):
        i = op_info[op]
        wrapper, wname = gen_wrapper(op, i["fn_type"], i["dispatcher"], i["ret_type"], i["args"])
        lines.append(wrapper)
        impl_lines.append(f'  m.impl("{op}", {wname});')
    lines.append("#endif  // FLAGOS_GEN_WRAPPERS")
    lines.append("")
    lines.append("// ---- m.impl() registrations ----")
    lines.append("#ifdef FLAGOS_GEN_IMPLS")
    lines += impl_lines
    lines.append("#endif  // FLAGOS_GEN_IMPLS")
    (out_dir / "register.inc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(op_info)} wrappers + impls")

    print("\nDone. Files in:", out_dir)


if __name__ == "__main__":
    main()
