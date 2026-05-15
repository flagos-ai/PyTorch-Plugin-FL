// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../index.h"

#include <ATen/ops/index.h>

#include "../../device_boxing.h"

namespace at::native::flagos {

namespace {

at::Tensor IndexKernelCuda(
    const at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices) {
  BoxToCuda(self);

  std::vector<at::Tensor> boxed_holders;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    auto opt = indices.get(i);
    if (opt.has_value() && opt->defined()) {
      BoxToCuda(*opt);
      boxed_holders.push_back(*opt);
    }
  }

  auto result = at::index(self, indices);

  UnboxToFlagos(self);
  for (auto& t : boxed_holders) {
    UnboxToFlagos(t);
  }
  UnboxToFlagos(result);
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(IndexTensorFn, index_tensor_stub, FlagosDevice::kCuda, IndexKernelCuda)

} // namespace at::native::flagos
