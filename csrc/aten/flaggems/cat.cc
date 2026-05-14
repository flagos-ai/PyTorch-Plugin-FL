// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/cat.h"

#include <flag_gems/operators.h>

namespace at::native::flagos {

namespace {

at::Tensor CatKernelFlaggems(const at::ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  std::vector<at::Tensor> tensor_vec(materialized.begin(), materialized.end());
  return flag_gems::cat(tensor_vec, dim);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(CatFn, cat_stub, FlagosDevice::kFlagOs, CatKernelFlaggems)

} // namespace at::native::flagos
