// Copyright (c) 2026, BAAI. All rights reserved.

#include "mul_scalar_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(MulScalarFn, mul_scalar_stub, "mul.Scalar")

namespace {

at::Tensor MulScalarKernelCuda(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.common_dtype(), "mul_scalar_cuda",
    [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      auto scalar_val = other.to<opmath_t>();
      at::native::gpu_kernel(iter, [scalar_val] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return static_cast<opmath_t>(a) * scalar_val;
      });
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MulScalarFn, mul_scalar_stub, FlagosDevice::kCuda, MulScalarKernelCuda)

} // namespace at::native::flagos
