// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/sin.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor SinKernelCuda(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.common_dtype(), "sin_cuda",
    [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::sin(a);
      });
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SinFn, sin_stub, FlagosDevice::kCuda, SinKernelCuda)

} // namespace at::native::flagos
