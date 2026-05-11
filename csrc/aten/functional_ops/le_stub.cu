// Copyright (c) 2026, BAAI. All rights reserved.

#include "le_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(LeTensorFn, le_tensor_stub, "le.Tensor")

namespace {

at::Tensor LeKernelCuda(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .add_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .declare_static_dtype(at::kBool)
    .build();
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.common_dtype(), "le_cuda", [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a <= b;
      });
    });
  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(LeTensorFn, le_tensor_stub, FlagosDevice::kCuda, LeKernelCuda)

} // namespace at::native::flagos
