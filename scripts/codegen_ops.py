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

        # Multi-output out-variants are now handled: gen_out_variant calls
        # at::<base>_outf(<all args faithful order>) and unboxes each mutable
        # Tensor& out, so no per-out-count skip is needed.

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


def python_kernel_name(fn_type: str) -> str:
    """AddTensorFn -> AddTensorKernelPython"""
    return fn_type[:-2] + 'KernelPython'


# ============================================================================
# FlagGems Python-path kernels (auto-discovered)
# ============================================================================
# Ops routed to the FlagGems Triton implementation via the embedded-Python
# caller (csrc/aten/backends/flagos/python_op_caller.{h,cc}). Rather than a
# hand-maintained map, discover_flaggems_ops() walks flag_gems._FULL_CONFIG and
# keeps only ops that can be wired SAFELY (see the arity/type gates below).
#
# Key facts (verified):
#   - flag_gems op names ARE aten schema names ("add.Tensor", "addmm.out",
#     "_softmax"), so no name mapping is needed.
#   - flag_gems functions take args by POSITION in aten faithful order (the
#     parameter *names* differ -- A/inp/self -- but positions align).
#   - Every generated kernel body packs the aten args (in faithful order) into a
#     std::vector<c10::IValue> and calls the generic caller; the func is located
#     by its "<module>.<name>" qualname.

# Types the generic IValue caller (python_op_caller.cc IValueToPython) can carry.
# ScalarType is deliberately EXCLUDED: an IValue stores it as a plain int, so it
# is indistinguishable from an ordinary int at runtime -> ops passing a dtype to
# the FlagGems function are dropped from the safe set (would silently mis-call).
_FLAGGEMS_GENERIC_OK = {
    "Tensor", "Tensor?", "Scalar", "Scalar?", "int", "int?",
    "float", "float?", "bool", "bool?", "SymInt", "SymInt?", "str", "str?",
}


def _flaggems_type_ok(t: str) -> bool:
    import re
    t = t.strip()
    if t in _FLAGGEMS_GENERIC_OK:
        return True
    if re.fullmatch(r"(int|SymInt|bool|float)\[\d*\]\??", t):
        return True
    if re.fullmatch(r"(int|SymInt|bool|float)\[\]\??", t):
        return True
    if t in ("Tensor[]", "Tensor?[]"):
        return True
    return False


def _flaggems_gems_npos(fn):
    """Number of positional params of a flag_gems function, or None if it has
    *args/**kwargs (uninspectable arity -> unsafe)."""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def _flaggems_kwonly_names(fn):
    """Set of keyword-only parameter names of a flag_gems function (params after
    the `*` in its signature), or empty set if uninspectable. These are the args
    gems accepts only by name -- the codegen forwards the matching aten trailing
    args as kwargs (dtype/alpha/correction/...)."""
    import inspect
    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return set()
    return {p.name for p in params if p.kind == p.KEYWORD_ONLY}


# aten types the keyword-arg forwarding path can express. ScalarType IS allowed
# here (unlike the positional _FLAGGEMS_GENERIC_OK set): the codegen tags a dtype
# kwarg so the caller converts it to a torch.dtype by name, sidestepping the
# "IValue stores ScalarType as int" ambiguity that blocks the positional path.
_FLAGGEMS_KWARG_OK = {
    "ScalarType", "ScalarType?", "Scalar", "Scalar?", "int", "int?",
    "float", "float?", "bool", "bool?", "str", "str?", "SymInt", "SymInt?",
}


def _flaggems_extra_trailing_ok(fn, ncall):
    """True if a gems function with more positional params than the `ncall` args
    we intend to pass can be safely called with exactly `ncall` positional args.

    Safe iff every gems param at index >= ncall has a default (so omitting it is
    fine) AND no such trailing param is named `out` sitting *before* a param we
    would still pass -- i.e. the extra params are strictly trailing, never
    interleaved. This rejects the reordering trap (e.g. gems gather inserts
    `out=None` at position 3, so aten's 4th arg `sparse_grad` would land in the
    `out` tensor slot). Returns False if uninspectable.
    """
    import inspect
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return False
    pos = [p for p in params
           if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(pos) < ncall:
        return False
    # Every gems param beyond the ones we pass must have a default.
    for p in pos[ncall:]:
        if p.default is inspect.Parameter.empty:
            return False
    # The first `ncall` gems params (which receive our aten args positionally)
    # must not include a param named `out`: that would mean gems expects `out`
    # among the leading positions and our aten arg would be misrouted into it.
    for p in pos[:ncall]:
        if p.name == "out":
            return False
    return True


# Categories the FlagGems Python path knows how to generate kernels for.
_FLAGGEMS_PY_CATEGORIES = {"functional_pure", "inplace", "tuple_return",
                           "out_variant", "factory"}

# TensorOptions field names carried by every factory schema after the shape/
# scalar positionals. The factory caller injects these itself (device=flagos,
# layout=strided, dtype forwarded, pin_memory=None), so they're stripped from
# the positional list the codegen passes. `memory_format` rides along on the
# *_like factories (like_factory caller injects it as None too).
_FLAGGEMS_TENSOROPT_FIELDS = {"dtype", "layout", "device", "pin_memory"}
_FLAGGEMS_LIKE_OPT_FIELDS = _FLAGGEMS_TENSOROPT_FIELDS | {"memory_format"}

# Random in-place ops we route to gems by injecting a CUDA generator (the
# RandomInplace caller). Whitelisted rather than type-detected because normal_
# is also inplace+Generator? but gems hardcodes generator=None internally (never
# forwards the one we pass -> IndexError on the empty PrivateUse1 default
# generator). Only ops verified to actually thread the generator through belong
# here. See the plan / [[flaggems-gap-analysis]].
_FLAGGEMS_RANDOM_INPLACE = {"uniform_", "exponential_", "bernoulli_.float"}

# Ops manually held out of the FlagGems Python path (populated during the
# compile/import/numerical convergence loop with the reason as a comment).
FLAGGEMS_PYTHON_SKIP = {
    # Required `out=` kwarg with no default: flag_gems `mm_out(a, b, *, out)`
    # forces out, but the generic positional caller cannot supply it, so the
    # kernel would raise. (7 other out-variants take out=None and are fine.)
    "mm.out",
    # Unconditional `assert X.device.type == device` where device == "cuda":
    # flagos tensors are genuinely PrivateUse1 ("privateuseone"), so the assert
    # can never pass without abandoning the boxing scheme. Op-specific to these.
    "maximum",
    "minimum",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "_upsample_bicubic2d_aa",
    # Same device assert ("Input tensor must be on CUDA device"): gems
    # _safe_softmax rejects the PrivateUse1 tensor before running.
    "_safe_softmax",
    # Random factories with NO generator parameter: gems' signature omits
    # `generator`, so its internal philox_backend_seed_offset(increment) reaches
    # for default_generators[device], but the PrivateUse1 device has no default
    # generator (IndexError); randperm also asserts an explicit int dtype. Cannot
    # be expressed without a per-device generator -- unlike uniform_/exponential_/
    # bernoulli_ which DO take a generator we inject (see _FLAGGEMS_RANDOM_INPLACE).
    # rand_like/randn_like share this root (same internal call, no generator arg);
    # they're rejected by the _like whitelist absence, listed here for the record.
    # See [[flaggems-gap-analysis]].
    "rand",
    "randn",
    "randperm",
    "rand_like",
    "randn_like",
    # normal family: gems' normal_ calls normal_distribution(..., generator=None)
    # with a hardcoded None, dropping the generator we pass, so it hits the empty
    # default_generators (IndexError). Upstream gems bug -- can't fix from here.
    "normal_",
    "normal.Tensor_Tensor",
    "normal.Tensor_float",
    "normal.float_Tensor",
}


def discover_flaggems_ops(codegen_ops, funcs):
    """Discover ops that can be safely routed to the FlagGems Python path.

    Returns {op_name: (gems_func_qualname, category, kwargs)} where kwargs is a
    list of (aten_type, name) for trailing aten args that gems declares
    keyword-only and the kernel forwards by name (dtype/alpha/correction/...);
    empty for the common all-positional case.

    Safety gates (see plan; validated in scratch analysis):
      - op must have a generated dispatcher (op in codegen_ops) and a schema.
      - flag_gems function arity must be inspectable (no *args/**kwargs).
      - category in {functional_pure, inplace, tuple_return, out_variant}.
      - ARITY gate (guards the silent "dropped trailing scalar" trap):
          functional_pure/inplace/tuple_return: gems npos == #aten args.
          out_variant: gems npos == #aten NON-out args (gems doesn't take `out`).
      - every arg passed to the gems function has a generic-caller-covered type.
    """
    try:
        # torch_fl activates the torch.cuda shim (CPU-torch reports no CUDA
        # otherwise), which flag_gems needs at import (get_device_name()). Must
        # run codegen through scripts/with_cuda_libtorch.sh so torch_fl imports.
        import torch_fl  # noqa: F401
        import flag_gems
    except Exception as e:
        print(f"  [flaggems] import failed ({e}); no python ops discovered",
              file=sys.stderr)
        return {}

    result = {}
    for item in flag_gems._FULL_CONFIG:
        if len(item) < 2:
            continue
        op, fn = item[0], item[1]
        if op in FLAGGEMS_PYTHON_SKIP:
            continue
        if op not in codegen_ops or op not in funcs:
            continue
        # Respect version-condition (item[2]) if present.
        if len(item) > 2 and callable(item[2]) and not item[2]():
            continue
        npos = _flaggems_gems_npos(fn)
        if npos is None:
            continue
        func = funcs[op]
        cat = detect_category(func)
        if cat not in _FLAGGEMS_PY_CATEGORIES:
            continue
        s = func.func
        aten_args = list(s.arguments.flat_all)
        out_args = list(s.arguments.out) if hasattr(s.arguments, "out") else []
        resolved_cat = cat
        # kwargs: aten trailing args gems declares keyword-only (dtype/alpha/...).
        # Filled by the arity_short branch below; each entry is (name, aten_type).
        kwargs = []
        # *_like factory: functional op whose schema is a source tensor (+ maybe
        # a value positional) followed by TensorOptions + memory_format, all
        # keyword-only in gems. The like_factory caller injects device=flagos/
        # layout/memory_format/pin_memory and forwards dtype; only the non-option
        # positionals are passed. gems reads shape/device from the source tensor.
        if cat == "functional_pure" and op.endswith("_like"):
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [(t, n) for t, n in all_pairs
                          if n not in _FLAGGEMS_LIKE_OPT_FIELDS]
            has_opts = any(n in _FLAGGEMS_LIKE_OPT_FIELDS for _t, n in all_pairs)
            byname = _flaggems_gems_byname_params(fn)
            # gems must accept dtype/layout/device by name and take exactly the
            # non-option positionals (the source tensor, plus full_like's
            # fill_value). npos may exceed via defaulted trailing kwonlys.
            if (not has_opts
                    or not {"dtype", "layout", "device"} <= byname
                    or npos < len(positional)
                    or (npos > len(positional)
                        and not _flaggems_extra_trailing_ok(fn, len(positional)))):
                continue
            if not all(_flaggems_type_ok(t) for t, _ in positional):
                continue
            qualname = f"{fn.__module__}.{fn.__name__}"
            result[op] = (qualname, "like_factory", positional)
            continue
        # Random in-place: whitelisted inplace op with a trailing Generator? arg.
        # The RandomInplace caller drops the generator positional and injects a
        # CUDA generator by name; only self + scalar positionals are passed.
        if cat == "inplace" and op in _FLAGGEMS_RANDOM_INPLACE:
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [(t, n) for t, n in all_pairs if t != "Generator?"]
            has_gen = any(t == "Generator?" for t, _n in all_pairs)
            # gems must take exactly self + the scalar params (generator is
            # keyword-only in gems, supplied by name).
            if (not has_gen
                    or npos < len(positional)
                    or (npos > len(positional)
                        and not _flaggems_extra_trailing_ok(fn, len(positional)))):
                continue
            if not all(_flaggems_type_ok(t) for t, _ in positional):
                continue
            qualname = f"{fn.__module__}.{fn.__name__}"
            result[op] = (qualname, "random_inplace", positional)
            continue
        if cat == "factory":
            # Factory: split off the TensorOptions fields (dtype/layout/device/
            # pin_memory); the factory caller injects device=flagos,
            # layout=strided, pin_memory=None and forwards dtype. Only the
            # shape/scalar positionals are passed. Require gems to declare
            # dtype/layout/device by name (kwonly on every gems factory) and its
            # positional count to match the non-option args.
            all_pairs = [(str(a.type), a.name) for a in aten_args]
            positional = [(t, n) for t, n in all_pairs
                          if n not in _FLAGGEMS_TENSOROPT_FIELDS]
            has_opts = any(n in _FLAGGEMS_TENSOROPT_FIELDS for _t, n in all_pairs)
            byname = _flaggems_gems_byname_params(fn)
            # gems must accept dtype/layout/device by name, and take exactly the
            # shape/scalar positionals (npos may exceed via a defaulted step, e.g.
            # arange.start's gems has an extra `step` default -> allow >=).
            if (not has_opts
                    or not {"dtype", "layout", "device"} <= byname
                    or npos < len(positional)
                    or (npos > len(positional)
                        and not _flaggems_extra_trailing_ok(fn, len(positional)))):
                continue
            if not all(_flaggems_type_ok(t) for t, _ in positional):
                continue
            qualname = f"{fn.__module__}.{fn.__name__}"
            result[op] = (qualname, "factory", positional)
            continue
        if cat == "out_variant":
            non_out = [(str(a.type), a.name) for a in aten_args if a not in out_args]
            with_out = non_out + [(str(a.type), a.name) for a in out_args]
            if npos == len(non_out) or (npos > len(non_out)
                                        and _flaggems_extra_trailing_ok(fn, len(non_out))):
                # gems takes only the non-out args (plus optional trailing
                # defaults), returns a fresh tensor; kernel copy_'s it into out.
                passed = non_out
            elif npos == len(with_out) or (npos > len(with_out)
                                           and _flaggems_extra_trailing_ok(fn, len(with_out))):
                # gems takes `out` positionally and writes into it; pass the out
                # tensor(s) too (plus optional trailing defaults like
                # memory_format=None). Distinguished from mm.out, whose gems out
                # is a required keyword-only arg the positional caller can't supply.
                passed = with_out
                resolved_cat = "out_variant_gemsout"
            elif npos < len(non_out):
                passed, kwargs = _flaggems_split_kwargs(fn, npos, non_out)
                if passed is None:
                    continue
            else:
                continue
        else:
            all_args = [(str(a.type), a.name) for a in aten_args]
            if npos == len(all_args) or (
                    npos > len(all_args) and _flaggems_extra_trailing_ok(fn, len(all_args))):
                passed = all_args
            elif npos < len(all_args):
                # gems takes fewer positional args: the trailing aten args must be
                # gems keyword-only params, forwarded by name (see kwarg path).
                passed, kwargs = _flaggems_split_kwargs(fn, npos, all_args)
                if passed is None:
                    continue
            else:
                continue
        # Promote ScalarType args out of the positional list into kwargs: the
        # positional caller can't carry a ScalarType (IValue stores it as a plain
        # int), but the by-name kwarg path tags it is_dtype and converts to a
        # torch.dtype. Safe only if gems accepts the arg BY NAME (a param with the
        # same name that isn't positional-only). This recovers ops like
        # _softmax_backward_data / linalg_vector_norm where gems takes dtype as a
        # positional-or-keyword param. See [[flaggems-gap-analysis]] type_gate_dtype.
        promotable = _flaggems_gems_byname_params(fn)
        promote_idx = [i for i, (t, name) in enumerate(passed)
                       if t.startswith("ScalarType") and name in promotable]
        # Only promote if the ScalarType args form a strict SUFFIX of the
        # positional list. Promoting a middle arg would shift the following
        # positionals into gems's dtype slot (wrong). All current ops have dtype
        # last, so this holds; the guard rejects any future middle-dtype op.
        if promote_idx and promote_idx == list(range(promote_idx[0], len(passed))):
            promoted = passed[promote_idx[0]:]
            passed = passed[:promote_idx[0]]
            kwargs = promoted + kwargs
        elif promote_idx:
            continue  # non-suffix ScalarType -> can't safely reorder
        if not all(_flaggems_type_ok(t) for t, _ in passed):
            continue
        if kwargs and not all(t in _FLAGGEMS_KWARG_OK for t, _ in kwargs):
            continue
        qualname = f"{fn.__module__}.{fn.__name__}"
        result[op] = (qualname, resolved_cat, kwargs)
    return result


def _flaggems_gems_byname_params(fn):
    """Names of gems params that can be passed BY NAME (keyword-only OR
    positional-or-keyword, i.e. anything except positional-only). Used to promote
    a ScalarType positional aten arg into a by-name kwarg the caller can convert
    to a torch.dtype."""
    import inspect
    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return set()
    return {p.name for p in params
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}


def _flaggems_split_kwargs(fn, npos, all_passed):
    """For an arity_short op (gems takes `npos` positional args, fewer than the
    `all_passed` aten args), split into (positional, kwargs). The trailing aten
    args beyond `npos` must ALL be gems keyword-only params matched BY NAME, else
    return (None, None) to reject the op (a name mismatch means we can't safely
    forward, and dropping the arg would silently change the result).

    Returns (positional_list, kwarg_list) where each list holds (aten_type, name)
    for positional and (aten_type, name) for kwargs.
    """
    positional = all_passed[:npos]
    trailing = all_passed[npos:]
    kwonly = _flaggems_kwonly_names(fn)
    for _t, name in trailing:
        if name not in kwonly:
            return None, None
    return positional, list(trailing)


def _flaggems_kwarg_cpp(kwargs):
    """Render the C++ `std::vector<PyKwarg>` initializer for the keyword args to
    forward to gems. Each PyKwarg is {name, value, is_dtype, is_none}; a dtype
    (ScalarType) arg is tagged is_dtype so the caller converts it to torch.dtype.
    Optional args (T?) are passed straight through -- the aten C++ arg is a
    std::optional, so a nullopt naturally maps to Python None inside the caller
    only if we mark it; but since we pass the IValue, an absent optional would
    become an int/None ambiguity for dtype. We therefore build the PyKwarg with a
    runtime `.has_value()` check for ScalarType? args."""
    parts = []
    for t, name in kwargs:
        is_dtype = t.startswith("ScalarType")
        if is_dtype:
            if t.endswith("?"):
                # std::optional<ScalarType>: None when absent, else tagged dtype.
                parts.append(
                    f'PyKwarg{{"{name}", {name}.has_value() ? '
                    f'c10::IValue(static_cast<int64_t>(*{name})) : c10::IValue(), '
                    f'/*is_dtype=*/true, /*is_none=*/!{name}.has_value()}}')
            else:
                parts.append(
                    f'PyKwarg{{"{name}", '
                    f'c10::IValue(static_cast<int64_t>({name})), '
                    f'/*is_dtype=*/true}}')
        else:
            # Scalar/int/bool/str/SymInt (and their optionals): IValue can carry
            # these directly. c10::IValue has implicit ctors for each.
            parts.append(f'PyKwarg{{"{name}", {name}}}')
    return "{" + ", ".join(parts) + "}"


def gen_flaggems_python_kernel(op, fn_type, ret_type, args, gems_func, category,
                               func, kwargs=None):
    """Generate a <Name>KernelPython forwarding to the FlagGems Python op.

    Signature matches the generated `*Fn` typedef exactly. The body packs the
    aten positional args (faithful order) into a std::vector<c10::IValue> and any
    keyword-only gems args (dtype/alpha/correction/...) into a
    std::vector<PyKwarg>, then calls the generic caller by qualname and adapts
    the result to the category's return convention:
      functional_pure -> UnboxToFlagos(result); return result;
      inplace         -> self.copy_(result); return self;  (or void)
      tuple_return    -> GenericTuple(...); unbox each; return make_tuple(...)
      out_variant     -> Generic/GenericTuple over NON-out args; out_i.copy_(res_i)
    """
    kn = python_kernel_name(fn_type)
    s = func.func
    aten_args = list(s.arguments.flat_all)
    out_args = list(s.arguments.out) if hasattr(s.arguments, "out") else []

    if category == "factory":
        # Factory: pass only the shape/scalar positionals; the factory caller
        # injects device=flagos/layout=strided/pin_memory=None and forwards
        # dtype. `kwargs` here carries the positional (aten_type, name) list.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        dtype_name = next(
            (a.name for a in aten_args if a.name == "dtype"), None)
        dtype_expr = dtype_name if dtype_name else "::std::nullopt"
        pos_init = "{" + ", ".join(pos_names) + "}"
        infer = ""
        # arange: gems defaults dtype=None to int64 unconditionally, but aten
        # infers float when ANY of start/end/step is floating. Left alone, gems
        # returns silently-wrong integer values for arange(0.,3.,.5). Replicate
        # aten's rule here so gems always gets an explicit dtype. Only arange has
        # this value-dependent inference; the other factories default to float32,
        # which gems already produces correctly.
        if op.startswith("arange"):
            scalar_names = [n for t, n in positional if "Scalar" in t]
            any_float = " || ".join(f"{n}.isFloatingPoint()" for n in scalar_names)
            infer = (
                f"  ::std::optional<at::ScalarType> _dt = {dtype_expr};\n"
                f"  if (!_dt.has_value()) _dt = ({any_float})\n"
                f"      ? at::typeMetaToScalarType(at::get_default_dtype()) "
                f": at::kLong;\n"
            )
            dtype_expr = "_dt"
        body = (
            f"{infer}"
            f'  auto result = CallPythonOp_Factory("{gems_func}", '
            f'{pos_init}, {dtype_expr});\n'
            f"  UnboxToFlagos(result);\n"
            f"  return result;"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    if category == "like_factory":
        # *_like: pass the source tensor (+ full_like's fill_value); the caller
        # injects device=flagos/layout/memory_format/pin_memory and forwards
        # dtype (nullopt -> None means "same as self"). `kwargs` here carries the
        # non-option positional (aten_type, name) list.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        dtype_name = next((a.name for a in aten_args if a.name == "dtype"), None)
        dtype_expr = dtype_name if dtype_name else "::std::nullopt"
        pos_init = "{" + ", ".join(pos_names) + "}"
        body = (
            f'  auto result = CallPythonOp_LikeFactory("{gems_func}", '
            f'{pos_init}, {dtype_expr});\n'
            f"  UnboxToFlagos(result);\n"
            f"  return result;"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    if category == "random_inplace":
        # Random in-place: pass self + scalar params (generator dropped); the
        # caller injects a CUDA generator by name. gems writes into self and
        # returns it; copy_ the result back to self for the aten alias contract.
        positional = kwargs or []
        pos_names = [n for _t, n in positional]
        self_name = pos_names[0]
        pos_init = "{" + ", ".join(pos_names) + "}"
        ret_line = "" if ret_type == "void" else f"\n  return {self_name};"
        body = (
            f'  auto result = CallPythonOp_RandomInplace("{gems_func}", '
            f'{pos_init});\n'
            f"  {self_name}.copy_(result);{ret_line}"
        )
        return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"

    kwargs = kwargs or []
    kw_names = {name for _t, name in kwargs}

    # arg names as they appear in the generated C++ signature. Kwarg-forwarded
    # args are removed from the positional list (they go into the PyKwarg vector).
    if category == "out_variant":
        passed_names = [a.name for a in aten_args
                        if a not in out_args and a.name not in kw_names]
    elif category == "out_variant_gemsout":
        # gems takes the out tensor(s) positionally after the non-out args.
        passed_names = ([a.name for a in aten_args
                         if a not in out_args and a.name not in kw_names]
                        + [a.name for a in out_args])
    else:
        passed_names = [a.name for a in aten_args if a.name not in kw_names]
    ivalues = "{" + ", ".join(passed_names) + "}"

    # kwarg call selects the *Kw caller variant; positional-only uses the plain one.
    if kwargs:
        kw_init = _flaggems_kwarg_cpp(kwargs)
        gen_call = f'CallPythonOp_GenericKw("{gems_func}", {ivalues}, {kw_init})'
        tuple_call = (lambda n:
                      f'CallPythonOp_GenericKwTuple("{gems_func}", {ivalues}, '
                      f'{kw_init}, {n})')
    else:
        gen_call = f'CallPythonOp_Generic("{gems_func}", {ivalues})'
        tuple_call = (lambda n:
                      f'CallPythonOp_GenericTuple("{gems_func}", {ivalues}, {n})')

    if category == "functional_pure":
        body = (
            f'  auto result = {gen_call};\n'
            f"  UnboxToFlagos(result);\n"
            f"  return result;"
        )
    elif category == "inplace":
        self_name = passed_names[0]
        ret_line = "" if ret_type == "void" else f"\n  return {self_name};"
        body = (
            f'  auto result = {gen_call};\n'
            f"  {self_name}.copy_(result);{ret_line}"
        )
    elif category == "tuple_return":
        n = ret_type.count(",") + 1  # ::std::tuple<a,b> -> 2
        unbox = "\n".join(f"  UnboxToFlagos(result[{i}]);" for i in range(n))
        make = ", ".join(f"result[{i}]" for i in range(n))
        body = (
            f'  auto result = {tuple_call(n)};\n'
            f"{unbox}\n"
            f"  return {{{make}}};"
        )
    elif category == "out_variant":
        out_names = [a.name for a in out_args]
        if ret_type.startswith("::std::tuple"):
            n = len(out_names)
            copies = "\n".join(
                f"  {out_names[i]}.copy_(result[{i}]);" for i in range(n)
            )
            make = ", ".join(out_names)
            body = (
                f'  auto result = {tuple_call(n)};\n'
                f"{copies}\n"
                f"  return {{{make}}};"
            )
        else:
            out_name = out_names[0]
            body = (
                f'  auto result = {gen_call};\n'
                f"  {out_name}.copy_(result);\n"
                f"  return {out_name};"
            )
    elif category == "out_variant_gemsout":
        # gems receives the out tensor(s) positionally and writes into them in
        # place, then returns them; just discard the returned handle and return
        # the aten out arg(s). (Only single-out ops reach here today.)
        out_name = [a.name for a in out_args][0]
        body = (
            f'  {gen_call};\n'
            f"  return {out_name};"
        )
    else:
        raise ValueError(f"unsupported flaggems-python category {category} for {op}")

    return f"{ret_type} {kn}({args_decl(args)}) {{\n{body}\n}}"


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
        # _foreach_*.out: TensorList out arg, call at::<base>_outf, void return.
        if s.is_out_fn():
            return "foreach_out"
        return "foreach_tensorlist"
    if s.is_out_fn():
        return "out_variant"
    if str(s.kind()).split('.')[-1] == "inplace":
        return "inplace"
    if len(s.returns) > 1:
        return "tuple_return"
    # split.Tensor / unbind.int: single Tensor[] return, no TensorList input.
    if len(s.returns) == 1 and str(s.returns[0].type) == "Tensor[]":
        return "vector_return"
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


def gen_functional_pure(op, fn_type, ret_type, args, func=None):
    kn = kernel_name(fn_type)
    guard = ", ".join(tensor_arg_names(args))
    api = f"at::{at_api_base(op)}"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  auto result = {api}({call_args(args)});
  UnboxToFlagos(result);
  return result;
}}"""


def gen_vector_return(op, fn_type, ret_type, args, func=None):
    """split.Tensor / unbind.int: single Tensor input, std::vector<Tensor> return
    (each element aliases the input storage). Box inputs, call the API, unbox
    every returned view."""
    kn = kernel_name(fn_type)
    guard = ", ".join(tensor_arg_names(args))
    api = f"at::{at_api_base(op)}"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  auto result = {api}({call_args(args)});
  UnboxTensorVecToFlagos(result);
  return result;
}}"""


def gen_inplace(op, fn_type, ret_type, args, func=None):
    """add_.Tensor / fill_.Scalar / masked_fill_.Scalar: mutate first tensor, return it (or void).

    Two calling conventions, selected by the op's torchgen `variants`:
      - method  (add_/masked_fill_): `self.add_(other, alpha)` -- these ops have
        NO free function, only a Tensor method.
      - function-only (silu_/gelu_/celu_/leaky_relu_/...activations): call the free
        function `at::silu_(self, ...)` -- these have NO Tensor method, so the
        method syntax fails to compile.
    Ops with both (fill_/zero_/relu_) work either way; we default to method."""
    kn = kernel_name(fn_type)
    tensors = tensor_arg_names(args)
    guard = ", ".join(tensors)
    base = at_api_base(op)  # e.g. "add_" already has trailing underscore
    self_name = args[0][1]
    other_args = ", ".join(n for _, n in args[1:])

    variants = [str(v).split('.')[-1] for v in func.variants] if func else []
    has_method = "method" in variants

    if has_method:
        # Method syntax: self.add_(other, alpha)
        body_call = (
            f"{self_name}.{base}({other_args});" if other_args
            else f"{self_name}.{base}();"
        )
    else:
        # Function-only syntax: at::silu_(self, ...)
        body_call = f"at::{base}({call_args(args)});"

    if ret_type == "void":
        ret_line = ""
    else:
        ret_line = f"\n  return {self_name};"
    return f"""{ret_type} {kn}({args_decl(args)}) {{
  DeviceBoxingGuard guard({guard});
  {body_call}{ret_line}
}}"""


def gen_out_variant(op, fn_type, ret_type, args, func=None):
    """Single- and multi-output out-variants (mm.out, sort.values, svd.U,
    native_batch_norm.out): call at::<base>_outf(<all args in faithful order>),
    which places the out args last -- exactly matching our generated arg order,
    so no reordering is needed. Then unbox each mutable Tensor& out.

    optional<Tensor> inputs must be materialized into a holder to be boxed by
    DeviceBoxingGuard (same pattern as gen_tuple_return)."""
    kn = kernel_name(fn_type)
    base = at_api_base(op)
    api = f"at::{base}_outf"

    # Mutable (non-const) Tensor& args are the outputs.
    out_names = [n for t, n in args if "at::Tensor &" in t and "const" not in t]

    # Box plain tensor inputs + optional<Tensor> inputs (via holders).
    plain = tensor_arg_names(args)
    opt_names = optional_tensor_names(args)
    holder_lines = ""
    guard_names = list(plain)
    for on in opt_names:
        holder_lines += f"  at::Tensor {on}_t = {on}.has_value() ? *{on} : at::Tensor();\n"
        guard_names.append(f"{on}_t")
    guard = ", ".join(guard_names)

    unbox_lines = "\n".join(f"  UnboxToFlagos({n});" for n in out_names)

    if ret_type.startswith("::std::tuple"):
        # tuple<Tensor&,...>: _ret references the out tensors; unbox out names.
        # (local is _ret, not result, because some ops have an out arg literally
        #  named `result`, e.g. _linalg_det.result.)
        return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
  auto _ret = {api}({call_args(args)});
{unbox_lines}
  return _ret;
}}"""
    # single mutable Tensor& out.
    out_name = out_names[0]
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
  {api}({call_args(args)});
{unbox_lines}
  return {out_name};
}}"""


def gen_tuple_return(op, fn_type, ret_type, args, func=None):
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


def gen_foreach(op, fn_type, ret_type, args, func=None):
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


def gen_foreach_out(op, fn_type, ret_type, args, func=None):
    """out-variants whose inputs include a TensorList. Two shapes:

      1. _foreach_*.out / split_copy.Tensor_out: the out is itself a
         `Tensor(a!)[]` list, return is void. Materialize + box every
         TensorList (inputs AND out list), call at::<base>_outf(...).
      2. cat.out / stack.out / block_diag.out / _chunk_cat.out: TensorList
         input(s) + a single mutable `Tensor(a!) out`, returning `Tensor&`.
         Box the input lists and the single out, call _outf, return the out.

    Args are already in faithful order (outs last), matching at::<base>_outf."""
    kn = kernel_name(fn_type)
    api = f"at::{at_api_base(op)}_outf"

    # Every mutable single Tensor& arg (non-list, non-const) must be boxed. For
    # void ops this includes an in-out like found_inf; for single-return ops it
    # is the lone `out`. Tuple-returning multi-Tensor& out RNN ops are skip-listed.
    mutable_tensors = [
        n for t, n in args
        if "at::Tensor &" in t and "const" not in t
        and "TensorList" not in t and "ITensorListRef" not in t
    ]

    materialize_lines = ""
    box_lines = ""
    call_arg_names = []
    for t, n in args:
        if "ITensorListRef" in t or "TensorList" in t:
            mat_name = f"{n}_vec"
            materialize_lines += f"  auto {mat_name} = MaterializeToTensorVec({n});\n"
            box_lines += f"  guard.box({mat_name});\n"
            call_arg_names.append(mat_name)
        else:
            if n in mutable_tensors:
                box_lines += f"  guard.box({{{n}}});\n"
            call_arg_names.append(n)
    call_args_str = ", ".join(call_arg_names)

    # Return shape follows ret_type: void (no return) or single `Tensor&`
    # (return the lone out). tuple<Tensor&,...> is not produced here (skipped).
    ret_line = ""
    if ret_type != "void":
        ret_line = f"\n  return {mutable_tensors[-1]};"

    return f"""{ret_type} {kn}({args_decl(args)}) {{
{materialize_lines}  TensorListBoxingGuard guard;
{box_lines}  {api}({call_args_str});{ret_line}
}}"""


def gen_factory(op, fn_type, ret_type, args, func=None):
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


def gen_optlist(op, fn_type, ret_type, args, func=None):
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
    "vector_return": gen_vector_return,
    "inplace": gen_inplace,
    "out_variant": gen_out_variant,
    "tuple_return": gen_tuple_return,
    "foreach_tensorlist": gen_foreach,
    "foreach_out": gen_foreach_out,
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

    # ---- discover FlagGems Python-path ops ----
    # Only meaningful in full-CUDA mode (all generated ops have dispatchers).
    # discover_flaggems_ops returns {op: (gems_func_qualname, category)}; keep
    # only ops that are actually in op_info (have a generated dispatcher slot).
    flaggems_py = discover_flaggems_ops(set(op_info), funcs)
    flaggems_py = {op: v for op, v in flaggems_py.items() if op in op_info}
    py_cat_counts = defaultdict(int)
    for _op, (_q, _c, _kw) in flaggems_py.items():
        py_cat_counts[_c] += 1
    print(f"\nFlagGems Python-path ops discovered: {len(flaggems_py)}")
    for cat in sorted(py_cat_counts):
        print(f"  {cat:20s} {py_cat_counts[cat]:3d} ops")

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
        lines.append(gen(op, i["fn_type"], i["ret_type"], i["args"], i["func"]))
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

    # ---- flaggems_python_kernels.cc ----
    # FlagGems Python-path kernels (Backend::kFlagOsPython slot) for the ops
    # auto-discovered by discover_flaggems_ops(). The whole file body is
    # guarded by FLAGOS_FLAGGEMS_PYTHON so it is a no-op unless the build sets
    # -DFLAGOS_FLAGGEMS_PYTHON (FLAGGEMS_PYTHON=ON), keeping non-flaggems builds
    # from pulling in python_op_caller / pybind.
    print("Generating flaggems_python_kernels.cc...")
    py_ops = [op for op in sorted(op_info) if op in flaggems_py]
    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "// AUTO-GENERATED by scripts/codegen_ops.py - DO NOT EDIT",
        "",
        "#ifdef FLAGOS_FLAGGEMS_PYTHON",
        "",
        "#include \"ops.h\"",
        "#include \"../device_boxing.h\"",
        "#include \"../backends/flagos/python_op_caller.h\"",
        "",
        "namespace at::native::flagos {",
        "namespace {",
        "",
    ]
    for op in py_ops:
        i = op_info[op]
        gems_func, category, kwargs = flaggems_py[op]
        lines.append(gen_flaggems_python_kernel(
            op, i["fn_type"], i["ret_type"], i["args"], gems_func, category,
            i["func"], kwargs))
        lines.append("")
    lines.append("} // namespace")
    lines.append("")
    for op in py_ops:
        i = op_info[op]
        pkn = python_kernel_name(i["fn_type"])
        lines.append(
            f'REGISTER_IMPL_TO_DISPATCHER({i["fn_type"]}, {i["dispatcher"]}, '
            f'Backend::kFlagOsPython, {pkn})'
        )
    lines.append("")
    lines.append("} // namespace at::native::flagos")
    lines.append("")
    lines.append("#endif  // FLAGOS_FLAGGEMS_PYTHON")
    (out_dir / "flaggems_python_kernels.cc").write_text("\n".join(lines) + "\n")
    print(f"   generated {len(py_ops)} flaggems-python kernels")

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

        # backends_flaggems.conf: same cuda routes, but the auto-discovered
        # FlagGems Python-path ops are flipped to flagos_python. Used for testing
        # / running the FlagGems path (FLAGOS_BACKEND_CONFIG=...backends_flaggems.conf).
        fg_conf_path = repo_root / "torch_fl/backends_flaggems.conf"
        fg_lines = [
            "# flagos op backend config -- AUTO-GENERATED (flaggems python mode)",
            "# Regenerated by scripts/codegen_ops.py with FLAGOS_CODEGEN_ALL=1.",
            "# Same as backends_cuda.conf, but the auto-discovered FlagGems ops are",
            "# routed to the flagos_python (kFlagOsPython) slot; all others to cuda.",
            "#",
            "# Format: op_name = backend   (backend: flaggems | flagos_python | cuda)",
            "",
        ]
        for op in sorted(op_info):
            backend = "flagos_python" if op in flaggems_py else "cuda"
            fg_lines.append(f"{op} = {backend}")
        fg_conf_path.write_text("\n".join(fg_lines) + "\n")
        print(f"   regenerated {fg_conf_path.name} with {len(flaggems_py)} flagos_python routes")

    print("\nDone. Files in:", out_dir)


if __name__ == "__main__":
    main()
