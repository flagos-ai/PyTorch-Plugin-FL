// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../topk.h"
#include "../../device_boxing.h"

#include <ATen/ops/topk.h>

namespace at::native::flagos {

namespace {

std::tuple<at::Tensor, at::Tensor> TopkKernelCuda(
    const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
  DeviceBoxingGuard guard(self);
  auto result = at::topk(self, k, dim, largest, sorted);
  UnboxToFlagos(std::get<0>(result));
  UnboxToFlagos(std::get<1>(result));
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(TopkFn, topk_dispatcher, Backend::kCuda, TopkKernelCuda)

} // namespace at::native::flagos
