// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../pow.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>

#include "native/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor PowTensorScalarKernelCuda(
    const at::Tensor& self, const at::Scalar& exp_scalar) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.common_dtype(), "pow_cuda",
    [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      const auto d_exp = static_cast<double>(exp);
      if (d_exp == 2.0) {
        at::native::gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base;
        });
      } else if (d_exp == 3.0) {
        at::native::gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base * base;
        });
      } else {
        at::native::gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base) -> scalar_t {
          return ::pow(base, exp);
        });
      }
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(PowTensorScalarFn, pow_tensor_scalar_stub, FlagosDevice::kCuda, PowTensorScalarKernelCuda)

} // namespace at::native::flagos
