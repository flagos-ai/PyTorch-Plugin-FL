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

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# TensorList C++ spelling (IListRef vs ArrayRef) must match what PyTorch itself
# registered for the op, or dispatcher registration crashes at import with
# "Mismatch in kernel C++ signatures".
#
# The authoritative rule is torchgen's own (torchgen/context.py):
#     use_ilistref_for_tensor_lists = f.part_of_structured_group
# i.e. an op that is part of a structured group uses IListRef (e.g. aten::cat);
# everything else (all _foreach_*, stack, _amp_foreach_*, ...) uses ArrayRef.
# Using this predicate instead of a hand-maintained set classifies ALL TensorList
# ops correctly in one shot, across every torch version.
def should_use_arrayref(func):
    """True -> emit ArrayRef; False -> emit IListRef. Mirrors torchgen's rule
    use_ilistref_for_tensor_lists = func.part_of_structured_group."""
    return not func.part_of_structured_group


# ============================================================================
# Full-CUDA enumeration mode (--all-cuda)
# ============================================================================
#
# Instead of reading a hand-maintained op list from backends_cuda.conf, the
# full mode enumerates EVERY leaf CUDA operator from native_functions.yaml and
# generates a boxing kernel for each. Ops the current templates cannot safely
# express are skipped and fall through to the existing cpu_fallback (functional,
# just slower), so coverage strictly grows and nothing regresses.
#
# Ops already registered by hand in csrc/aten/register.cc. Generating these
# would collide at registration (duplicate m.impl) -> MUST be excluded.
MANUAL_REGISTERED_OPS = {
    "empty.memory_format",
    "empty_strided",
    "as_strided",
    "resize_",
    "_reshape_alias",
    "_copy_from",
    "_copy_from_and_resize",
    "copy_",
    "_local_scalar_dense",
    "set_.source_Tensor",
    "set_.source_Storage",
    "set_.source_Storage_storage_offset",
    "view",
    "contiguous",
    "clone",
    "_to_copy",
    "index_put_",
    "_index_put_impl_",
    "record_stream",
}


def _delegate_target_has_cuda(func, funcs, cuda_index):
    """A structured_delegate op routes its CUDA kernel through the named target
    (e.g. add.Tensor -> add.out). Return True if that target has a CUDA kernel."""
    deleg = getattr(func, "structured_delegate", None)
    if deleg is None:
        return False
    target = funcs.get(str(deleg))
    return target is not None and cuda_index.has_kernel(target)


def cuda_supported(func, funcs, cuda_index):
    """
    True if this op can be executed on CUDA (directly or by decomposition),
    matching the union that makes the enumeration a superset of the existing
    hand-written 71-op conf:
      1. direct CUDA kernel (functional/out leaf), OR
      2. structured_delegate whose target has a CUDA kernel (add/mm/silu_backward
         route their kernel through the .out / .grad_input variant), OR
      3. CompositeExplicitAutograd kernel (sort/abs/div.Scalar decompose into
         CUDA sub-ops).
    structured_delegate is checked BEFORE the composite_implicit exclusion so
    ops that carry both flags (e.g. silu_backward) are not dropped.
    """
    if cuda_index.has_kernel(func):
        return True
    if _delegate_target_has_cuda(func, funcs, cuda_index):
        return True
    if func.has_composite_explicit_autograd_kernel:
        return True
    return False


def enumerate_all_cuda_ops(nf, funcs, cuda_index):
    """
    Returns (kept_ops, skipped) where kept_ops is the list of op-name strings to
    generate and skipped is {reason: [op, ...]}.

    Skip reasons:
      manual      already registered by hand in register.cc
      multi_out   out-variant with >1 mutable Tensor& out (template picks wrong out)
    composite_implicit ops are excluded up front: PyTorch decomposes them ABOVE
    our dispatch key into leaf ops we already box, so registering them is both
    unnecessary and risky. structured_delegate ops survive that exclusion.
    """
    kept = []
    skipped = defaultdict(list)
    for func in nf.native_functions:
        op = str(func.func.name)

        if not cuda_supported(func, funcs, cuda_index):
            continue

        # composite_implicit (and NOT structured_delegate) -> decomposed above us
        if (func.has_composite_implicit_autograd_kernel
                and getattr(func, "structured_delegate", None) is None
                and not func.has_composite_explicit_autograd_kernel):
            continue

        if op in MANUAL_REGISTERED_OPS:
            skipped["manual"].append(op)
            continue

        # multi-out: gen_out_variant assumes exactly one mutable Tensor& out
        if func.func.is_out_fn():
            n_out = sum(
                1 for a in func.func.arguments.flat_all
                if "Tensor" in str(a.type) and a.is_write
            )
            if n_out > 1:
                skipped["multi_out"].append(op)
                continue

        kept.append(op)
    return kept, skipped

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
    stripped = base.lstrip('_')
    n_lead = len(base) - len(stripped)  # leading underscores: _conv vs conv
    is_inplace = stripped.endswith('_')
    base_clean = stripped.rstrip('_') if is_inplace else stripped

    # Disambiguation token for leading underscores (private/internal ops).
    # foreach ops keep the conventional single leading '_' implicit (unique via
    # the "Foreach" prefix / "foreach_" core), so they are exempt to keep the
    # 71-op names stable. Every other leading underscore is preserved so
    # `_convolution` and `convolution` (both leaf CUDA ops) do not collide.
    priv_pascal = "" if is_foreach else "Priv" * n_lead
    priv_snake = "" if is_foreach else "priv_" * n_lead

    # --- PascalCase type name ---
    if is_foreach:
        core = base_clean[len('foreach_'):]
        type_base = 'Foreach' + ''.join(w.capitalize() for w in core.split('_') if w)
    else:
        type_base = ''.join(w.capitalize() for w in base_clean.split('_') if w)
    if is_inplace:
        type_base += 'Inplace'
    # A trailing underscore on the variant (e.g. "out_") marks a mutating variant
    # distinct from the non-mutating one ("out"); preserve it so range.out and
    # range.out_ do not collapse to the same name.
    variant_mut = variant.endswith('_') if variant else False
    if variant:
        type_base += ''.join(w.capitalize() for w in variant.split('_') if w)
    if variant_mut:
        type_base += 'Mut'
    fn_type = priv_pascal + type_base + 'Fn'

    # --- snake_case dispatcher name ---
    disp = priv_snake + base_clean  # foreach already normalized (leading _ stripped)
    if is_inplace:
        disp += '_inplace'
    if variant:
        disp += '_' + variant.lower().rstrip('_')
    if variant_mut:
        disp += '_mut'
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
    size_arg = names[1] if has_self else (names[0] if names else "{}")

    # --- fill-allocators: our own at::empty allocator + a fill. No CUDA compute
    #     kernel, no recursion (at::empty is hand-registered as the real allocator). ---
    if base in ("zeros", "new_zeros"):
        make = f"  auto result = at::empty({size_arg}, options);\n  result.zero_();"
    elif base in ("ones", "new_ones"):
        make = f"  auto result = at::empty({size_arg}, options);\n  result.fill_(1);"
    elif base in ("full", "new_full"):
        # fill value is the lone by-value Scalar arg (not Scalar[] / optional<Scalar>)
        fill_val = next(
            (n for t, n in args
             if "Scalar" in t and "ArrayRef" not in t and "optional" not in t),
            "0",
        )
        make = f"  auto result = at::empty({size_arg}, options);\n  result.fill_({fill_val});"
    elif base == "scalar_tensor":
        make = f"  auto result = at::empty({{}}, options);\n  result.fill_({names[0]});"
    else:
        # --- compute factories (arange/rand/randn/randint/randperm/normal/eye/
        #     linspace/logspace/*_window/fft_*freq/tril_indices/...): must run the
        #     real kernel. Calling it with a PrivateUse1 device re-dispatches into
        #     THIS kernel -> infinite recursion -> stack overflow. Redirect the
        #     device arg to CUDA (hits the external libtorch_cuda.so kernel), then
        #     unbox the result back to flagos. Generalizes the old arange special-case. ---
        device_arg = next(
            (n for t, n in args if "optional<at::Device>" in t), None
        )
        if device_arg is not None:
            call_names = [
                "::std::optional<at::Device>(_cuda_dev)" if n == device_arg else n
                for _, n in args
            ]
            make = (
                f"  at::Device _req_dev = {device_arg}.has_value() ? *{device_arg} "
                f": at::Device(at::kPrivateUse1, 0);\n"
                "  at::Device _cuda_dev = _req_dev.type() == at::kPrivateUse1\n"
                "      ? at::Device(at::kCUDA, _req_dev.index()) : _req_dev;\n"
                f"  auto result = at::{base}({', '.join(call_names)});\n"
                "  if (result.device().type() == at::kCUDA) UnboxToFlagos(result);"
            )
        else:
            # no device knob -> best-effort empty allocation
            make = f"  auto result = at::empty({size_arg}, options);"

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

def api_headers(op_info: Dict) -> List[str]:
    """Header file names come from torchgen's authoritative func.root_name.
    (e.g. schema '__ilshift__.Scalar' -> root_name 'lshift' -> ATen/ops/lshift.h;
    'add.out' and 'add.Tensor' both -> 'add'.) Never derive the header from the
    schema base by stripping underscores -- that mangles dunder operators."""
    bases = set()
    for op in op_info.values():
        bases.add(op["func"].root_name)
    return [f"#include <ATen/ops/{b}.h>" for b in sorted(bases)]


# ============================================================================
# Main
# ============================================================================

def main():
    repo_root = Path(__file__).parent.parent
    conf_path = repo_root / "torch_fl/backends_cuda.conf"
    out_dir = repo_root / "csrc/aten/generated"
    out_dir.mkdir(exist_ok=True)

    print("Loading configuration and schemas...")

    root = Path(torchgen.__file__).parent
    nf = parse_native_yaml(
        str(root / "packaged/ATen/native/native_functions.yaml"),
        str(root / "packaged/ATen/native/tags.yaml"),
    )
    funcs = {str(f.func.name): f for f in nf.native_functions}

    # Ops the templates cannot compile yet; skipped -> fall through to cpu_fallback.
    # Grown empirically during the compile/import fix loop.
    skip_ops_path = repo_root / "torch_fl/codegen_skip_ops.txt"
    manual_skip = set()
    if skip_ops_path.exists():
        for line in skip_ops_path.read_text().splitlines():
            line = line.split('#')[0].strip()
            if line:
                manual_skip.add(line)

    all_cuda = os.environ.get("FLAGOS_CODEGEN_ALL", "").strip() not in ("", "0", "false")

    if all_cuda:
        from torchgen.model import DispatchKey
        cuda_index = nf.backend_indices[DispatchKey.CUDA]
        ops, skipped = enumerate_all_cuda_ops(nf, funcs, cuda_index)
        ops = [o for o in ops if o not in manual_skip]
        n_manual_skip = len(manual_skip)
        print(f"[FULL CUDA MODE] enumerated {len(ops)} ops to generate")
        for reason in sorted(skipped):
            print(f"   skipped[{reason}]: {len(skipped[reason])}")
        print(f"   skipped[template_skip_list]: {n_manual_skip}")
    else:
        ops = []
        for line in conf_path.read_text().splitlines():
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            op, backend = line.split('=', 1)
            if backend.strip() == "cuda":
                ops.append(op.strip())
        print(f"Found {len(ops)} ops in backends_cuda.conf")

    op_info = {}
    categories = defaultdict(list)

    # Generate signatures per-operator with correct IListRef/ArrayRef setting
    for op in ops:
        if op not in funcs:
            print(f"  WARNING: {op} not in native_functions.yaml", file=sys.stderr)
            continue
        func = funcs[op]
        try:
            cat = detect_category(func)
            fn_type, dispatcher = schema_to_cpp_name(op)

            # Determine if this op needs ArrayRef (vs IListRef default)
            use_arrayref = should_use_arrayref(func)

            # Generate signature with operator-specific IListRef/ArrayRef and
            # const-ref-for-mutable-tensors settings, both taken from torchgen's
            # authoritative per-op flags (matches PyTorch's own registration).
            with local.parametrize(
                use_const_ref_for_mutable_tensors=func.use_const_ref_for_mutable_tensors,
                use_ilistref_for_tensor_lists=not use_arrayref,  # False=ArrayRef, True=IListRef
            ):
                # Use unified CppSignature (faithful) for typedef, kernel, AND wrapper
                ptr_type, ret_type, args = unified_sig(func)
        except Exception as e:
            if all_cuda:
                print(f"  SKIP {op}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                continue
            raise

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
    lines += api_headers(op_info)
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

    # In full mode, regenerate backends_cuda.conf so GetBackendForOp routes every
    # generated op to cuda. Ops NOT in op_info (skipped) are absent -> they hit
    # the PrivateUse1 cpu_fallback, exactly as before this change.
    if all_cuda:
        conf_lines = [
            "# flagos op backend config -- AUTO-GENERATED (full CUDA mode)",
            "# Regenerated by scripts/codegen_ops.py with FLAGOS_CODEGEN_ALL=1.",
            "# Every generated boxing kernel is routed to the cuda backend; ops not",
            "# listed here are not registered and fall through to cpu_fallback.",
            "#",
            "# Format: op_name = backend   (backend: flaggems | flagos_python | cuda)",
            "",
        ]
        for op in sorted(op_info):
            conf_lines.append(f"{op} = cuda")
        conf_path.write_text("\n".join(conf_lines) + "\n")
        print(f"   regenerated {conf_path.name} with {len(op_info)} cuda routes")

    print("\nDone. Files in:", out_dir)


if __name__ == "__main__":
    main()
