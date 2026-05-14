// Copyright (c) 2026, BAAI. All rights reserved.

#include "div_scalar.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(DivScalarFn, div_scalar_stub, "div.Scalar")

namespace {

at::Tensor DivScalarKernelCuda(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.common_dtype(), "div_scalar_cuda",
    [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      auto inv = static_cast<opmath_t>(1.0) / other.to<opmath_t>();
      at::native::gpu_kernel(iter, [inv] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return static_cast<opmath_t>(a) * inv;
      });
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(DivScalarFn, div_scalar_stub, FlagosDevice::kCuda, DivScalarKernelCuda)

} // namespace at::native::flagos
