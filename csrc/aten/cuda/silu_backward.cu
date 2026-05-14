// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/silu_backward.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor SiluBackwardKernelCuda(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(grad_output)
    .add_input(self)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.dtype(), "silu_backward_cuda",
    [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t grad, scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t x_acc = static_cast<opmath_t>(x);
        const opmath_t sigmoid = opmath_t(1) / (opmath_t(1) + ::exp(-x_acc));
        return static_cast<opmath_t>(grad) * sigmoid * (opmath_t(1) + x_acc * (opmath_t(1) - sigmoid));
      });
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SiluBackwardFn, silu_backward_stub, FlagosDevice::kCuda, SiluBackwardKernelCuda)

} // namespace at::native::flagos
