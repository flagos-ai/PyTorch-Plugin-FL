// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../index.h"

#include <ATen/ops/index.h>

namespace at::native::flagos {

namespace {

at::Tensor IndexKernelMetax(
    const at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices) {
  at::Tensor self_cpu = self.cpu();

  c10::List<::std::optional<at::Tensor>> indices_cpu;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    auto opt = indices.get(i);
    if (opt.has_value() && opt->defined()) {
      indices_cpu.push_back(opt->cpu());
    } else {
      indices_cpu.push_back(std::nullopt);
    }
  }

  at::Tensor result_cpu = at::index(self_cpu, indices_cpu);
  return result_cpu.to(self.device(), result_cpu.scalar_type());
}

}  // namespace

FLAGOS_REGISTER_DISPATCH(
    IndexTensorFn, index_tensor_stub, FlagosDevice::kMetax, IndexKernelMetax)

}  // namespace at::native::flagos
