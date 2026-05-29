// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cat.h"

#include <flag_gems/operators.h>

namespace at::native::flagos {

namespace {

at::Tensor CatKernelFlaggems(const at::ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  std::vector<at::Tensor> tensor_vec(materialized.begin(), materialized.end());
  return flag_gems::cat(tensor_vec, dim);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kFlagOs, CatKernelFlaggems)

} // namespace at::native::flagos
