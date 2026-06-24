// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add_inplace.h"
#include "../../device_boxing.h"

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>

#include "native/Loops.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
struct AddInplaceCudaFunctorOnSelf {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t other_;
  opmath_t alpha_;
  AddInplaceCudaFunctorOnSelf(opmath_t other, opmath_t alpha) : other_(other), alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t self) const {
    return static_cast<opmath_t>(self) + alpha_ * other_;
  }
};

template <typename scalar_t>
struct AddInplaceCudaFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha_;
  AddInplaceCudaFunctor(opmath_t alpha) : alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t self, scalar_t other) const {
    return static_cast<opmath_t>(self) + alpha_ * static_cast<opmath_t>(other);
  }
};

void AddInplaceKernelCuda(
    at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  BoxToCuda(self);
  BoxToCuda(other);

  auto iter = at::TensorIteratorConfig()
    .add_output(self)
    .add_input(self)
    .add_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    iter.common_dtype(), "add_inplace_cuda",
    [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      if (iter.is_cpu_scalar(2)) {
        AddInplaceCudaFunctorOnSelf<scalar_t> ufunctor(
          iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>());
        iter.remove_operand(2);
        at::native::gpu_kernel(iter, ufunctor);
      } else {
        at::native::gpu_kernel(iter, AddInplaceCudaFunctor<scalar_t>(alpha.to<opmath_t>()));
      }
    }
  );

  UnboxToFlagos(self);
  UnboxToFlagos(other);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AddInplaceTensorFn, add_inplace_tensor_dispatcher, Backend::kCuda, AddInplaceKernelCuda)

} // namespace at::native::flagos
