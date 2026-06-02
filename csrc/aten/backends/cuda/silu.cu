// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../silu.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "native/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor SiluKernelCuda(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.dtype(), "silu_cuda",
    [&]() {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t x_acc = static_cast<opmath_t>(x);
        return x_acc / (opmath_t(1) + ::exp(-x_acc));
      });
    }
  );

  return iter.output();
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SiluFn, silu_dispatcher, Backend::kCuda, SiluKernelCuda)

} // namespace at::native::flagos
