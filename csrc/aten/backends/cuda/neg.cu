// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../neg.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "native/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor NegKernelCuda(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.common_dtype(), "neg_cuda",
    [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return -a;
      });
    }
  );

  return iter.output();
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(NegFn, neg_dispatcher, Backend::kCuda, NegKernelCuda)

} // namespace at::native::flagos
