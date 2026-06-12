// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../multinomial.h"
#include "../../device_boxing.h"

#include <ATen/ops/multinomial.h>

namespace at::native::flagos {

namespace {

at::Tensor MultinomialKernelCuda(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    ::std::optional<at::Generator> generator) {
  DeviceBoxingGuard guard(self);
  auto result = at::multinomial(self, num_samples, replacement, generator);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MultinomialFn, multinomial_dispatcher, Backend::kCuda, MultinomialKernelCuda)

} // namespace at::native::flagos
