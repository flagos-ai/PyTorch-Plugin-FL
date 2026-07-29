// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// multinomial(Tensor self, int num_samples, bool replacement=False, *,
//             Generator? generator=None) -> Tensor
//
// aclnnMultinomial(self, numsamples, replacement, seed, offset, out). Output is
// int64 sampled indices with the sample dim replaced by num_samples ([N] input
// -> [num_samples]; [B, N] -> [B, num_samples]). transformers' _sample() calls
// this to draw the next token. We pull a 64-bit seed from the default CPU
// generator so successive calls differ; offset is left at 0.
at::Tensor MultinomialKernelAscend(const at::Tensor& self, int64_t num_samples,
                                   bool replacement,
                                   ::std::optional<at::Generator> generator) {
  namespace ascend = at::native::flagos::ascend;

  auto out_shape = self.sizes().vec();
  out_shape.back() = num_samples;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  // Derive a seed. Prefer the supplied generator, else the default one.
  at::Generator gen = generator.has_value()
      ? generator.value()
      : at::detail::getDefaultCPUGenerator();
  int64_t seed;
  {
    std::lock_guard<std::mutex> lock(gen.mutex());
    seed = static_cast<int64_t>(
        at::check_generator<at::CPUGeneratorImpl>(gen)->random64());
  }
  int64_t offset = 0;

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnMultinomial, acl_self.get(), num_samples, replacement,
                  seed, offset, acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(MultinomialFn, multinomial_dispatcher, Backend::kAscend, MultinomialKernelAscend)

} // namespace at::native::flagos
