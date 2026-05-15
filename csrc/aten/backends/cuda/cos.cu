// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cos.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "native/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor CosKernelCuda(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.common_dtype(), "cos_cuda",
    [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::cos(a);
      });
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(CosFn, cos_stub, FlagosDevice::kCuda, CosKernelCuda)

} // namespace at::native::flagos
