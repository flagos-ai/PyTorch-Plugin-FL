// Copyright (c) 2026, BAAI. All rights reserved.

#include "rsqrt_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(RsqrtFn, rsqrt_stub, "rsqrt")

namespace {

at::Tensor rsqrt_kernel_cuda(const at::Tensor& self) {
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

FLAGOS_REGISTER_DISPATCH(RsqrtFn, rsqrt_stub, FlagosDevice::kCuda, rsqrt_kernel_cuda)

} // namespace at::native::flagos
