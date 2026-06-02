// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../sum.h"

namespace at::native::flagos {

namespace {

at::Tensor SumDimKernelPython(const at::Tensor& self, at::OptionalIntArrayRef dim,
                              bool keepdim, std::optional<at::ScalarType> dtype) {
  // When dim is None or empty, this is a full reduction.
  // Call FlagGems' "sum" directly (uses triton kernels, no recursion).
  // Cannot call "sum_dim" with dim=None: it calls torch.sum() → re-dispatch loop.
  // Cannot pass dim=[] as tuple: FlagGems checks `if dim == []` (list), () != [].
  bool is_full_reduce = !dim.has_value() || dim->empty();
  if (is_full_reduce) {
    at::Tensor out = CallPythonOp_TD("sum", self, dtype);
    if (keepdim) {
      std::vector<int64_t> shape(self.dim(), 1);
      out = out.reshape(shape);
    }
    return out;
  }
  return CallPythonOp_TOIB("sum_dim", self, dim, keepdim, dtype);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SumDimFn, sum_dim_dispatcher, Backend::kFlagOsPython, SumDimKernelPython)

} // namespace at::native::flagos
