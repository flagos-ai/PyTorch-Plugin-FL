// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../all.h"

#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <ATen/native/cuda/Reduce.cuh>

namespace at::native::flagos {

namespace {

at::Tensor AllKernelCuda(const at::Tensor& self) {
  at::Tensor result = at::empty({}, self.options().dtype(at::kBool));

  if (self.numel() == 0) {
    // Empty tensor: all() is vacuously true
    result.fill_(true);
    return result;
  }

  auto iter = at::TensorIterator::reduce_op(result, self.to(at::kBool));

  at::native::gpu_reduce_kernel<bool, bool>(
      iter,
      func_wrapper<bool>([] GPU_LAMBDA(bool acc, bool val) -> bool {
        return acc && val;
      }),
      true);

  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AllFn, all_stub, FlagosDevice::kCuda, AllKernelCuda)

} // namespace at::native::flagos
