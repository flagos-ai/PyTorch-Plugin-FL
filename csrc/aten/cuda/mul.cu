// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/mul.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

namespace {

template <typename T>
struct MulFunctor {
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

template <>
struct MulFunctor<bool> {
  __device__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

at::Tensor MulKernelCuda(
    const at::Tensor& self, const at::Tensor& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .add_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.common_dtype(), "mul_cuda",
    [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      at::native::opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, MulFunctor<opmath_t>());
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MulTensorFn, mul_tensor_stub, FlagosDevice::kCuda, MulKernelCuda)

} // namespace at::native::flagos
