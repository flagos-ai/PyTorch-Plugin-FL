// Copyright (c) 2026, BAAI. All rights reserved.

#include "add_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(AddTensorFn, add_tensor_stub, "add.Tensor")

namespace {

// CUDA functors for add operation
template <typename scalar_t>
struct AddCudaFunctorOnSelf {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t other_;
  opmath_t alpha_;
  AddCudaFunctorOnSelf(opmath_t other, opmath_t alpha) : other_(other), alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t self) const {
    return static_cast<opmath_t>(self) + alpha_ * other_;
  }
};

template <typename scalar_t>
struct AddCudaFunctorOnOther {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t self_;
  opmath_t alpha_;
  AddCudaFunctorOnOther(opmath_t self, opmath_t alpha) : self_(self), alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t other) const {
    return self_ + alpha_ * static_cast<opmath_t>(other);
  }
};

template <typename scalar_t>
struct AddCudaFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha_;
  AddCudaFunctor(opmath_t alpha) : alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t self, scalar_t other) const {
    return static_cast<opmath_t>(self) + alpha_ * static_cast<opmath_t>(other);
  }
};

at::Tensor add_kernel_cuda(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
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

  // Dispatch based on dtype and call gpu_kernel with our modified Loops.cuh
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.common_dtype(), "add_cuda",
    [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      if (iter.is_cpu_scalar(1)) {
        AddCudaFunctorOnOther<scalar_t> ufunctor(
          iter.scalar_value<opmath_t>(1), alpha.to<opmath_t>());
        iter.remove_operand(1);
        at::native::gpu_kernel(iter, ufunctor);
      } else if (iter.is_cpu_scalar(2)) {
        AddCudaFunctorOnSelf<scalar_t> ufunctor(
          iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>());
        iter.remove_operand(2);
        at::native::gpu_kernel(iter, ufunctor);
      } else {
        at::native::gpu_kernel(iter, AddCudaFunctor<scalar_t>(alpha.to<opmath_t>()));
      }
    }
  );

  return iter.output();
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AddTensorFn, add_tensor_stub, FlagosDevice::kCuda, add_kernel_cuda)

} // namespace at::native::flagos
