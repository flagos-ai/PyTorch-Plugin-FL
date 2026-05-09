// Copyright (c) 2026, BAAI. All rights reserved.

#include "all_stub.h"

#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <ATen/native/cuda/Reduce.cuh>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(AllFn, all_stub, "all")

namespace {

at::Tensor AllKernelCuda(const at::Tensor& self) {
  at::Tensor result = at::empty({}, self.options().dtype(at::kBool));

  if (self.numel() == 0) {
    // Empty tensor: all() is vacuously true
    result.fill_(true);
    return result;
  }

  auto iter = at::TensorIterator::reduce_op(result, self.to(at::kBool));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
    self.scalar_type(), "all_cuda",
    [&]() {
      at::native::gpu_reduce_kernel<scalar_t, bool>(
          iter,
          func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
            return acc && static_cast<bool>(val);
          }),
          true);
    }
  );

  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AllFn, all_stub, FlagosDevice::kCuda, AllKernelCuda)

} // namespace at::native::flagos
