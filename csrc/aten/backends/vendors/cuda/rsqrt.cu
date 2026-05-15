// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../../rsqrt.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "native/cuda/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor RsqrtKernelCuda(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::BFloat16, at::ScalarType::Half,
    iter.common_dtype(), "rsqrt_cuda",
    [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::rsqrt(a);
      });
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(RsqrtFn, rsqrt_stub, FlagosDevice::kCuda, RsqrtKernelCuda)

} // namespace at::native::flagos
