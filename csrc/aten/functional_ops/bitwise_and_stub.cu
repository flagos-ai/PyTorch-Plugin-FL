// Copyright (c) 2026, BAAI. All rights reserved.

#include "bitwise_and_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(BitwiseAndTensorFn, bitwise_and_tensor_stub, "bitwise_and.Tensor")

namespace {

at::Tensor BitwiseAndKernelCuda(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .add_input(other)
    .build();
  AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.common_dtype(), "bitwise_and_cuda", [&]() {
    at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a & b;
    });
  });
  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(BitwiseAndTensorFn, bitwise_and_tensor_stub, FlagosDevice::kCuda, BitwiseAndKernelCuda)

} // namespace at::native::flagos
