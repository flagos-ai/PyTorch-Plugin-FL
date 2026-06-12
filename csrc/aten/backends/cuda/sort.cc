// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sort.h"
#include "../../device_boxing.h"

#include <ATen/ops/sort.h>

namespace at::native::flagos {

namespace {

std::tuple<at::Tensor, at::Tensor> SortKernelCuda(
    const at::Tensor& self, int64_t dim, bool descending) {
  DeviceBoxingGuard guard(self);
  auto result = at::sort(self, dim, descending);
  UnboxToFlagos(std::get<0>(result));
  UnboxToFlagos(std::get<1>(result));
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SortFn, sort_dispatcher, Backend::kCuda, SortKernelCuda)

} // namespace at::native::flagos
