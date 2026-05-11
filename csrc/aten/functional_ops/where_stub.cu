// Copyright (c) 2026, BAAI. All rights reserved.

#include "where_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>

#include "../native/cuda/Loops.cuh"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(WhereSelfFn, where_self_stub, "where.self")

namespace {

at::Tensor WhereKernelCuda(
    const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
  auto result_type = at::result_type(self, other);
  auto self_cast = self.to(result_type);
  auto other_cast = other.to(result_type);

  auto result_shape = at::infer_size(condition.sizes(),
                                     at::infer_size(self_cast.sizes(), other_cast.sizes()));
  at::Tensor output = at::empty(result_shape, self_cast.options());

  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(condition)
    .add_input(self_cast)
    .add_input(other_cast)
    .check_all_same_dtype(false)
    .build();

  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    result_type, "where_cuda", [&]() {
      at::native::gpu_kernel(iter,
        [] GPU_LAMBDA(bool cond, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond ? self_val : other_val;
        });
    });
  return output;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(WhereSelfFn, where_self_stub, FlagosDevice::kCuda, WhereKernelCuda)

} // namespace at::native::flagos
