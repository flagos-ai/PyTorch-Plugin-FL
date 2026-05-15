// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../../sum.h"

#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <ATen/native/cuda/Reduce.cuh>

namespace at::native::flagos {

namespace {

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
void SumKernelImpl(at::TensorIterator& iter) {
  constexpr bool is_16_bits = sizeof(scalar_t) == 2;
  if constexpr (is_16_bits) {
    at::native::gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, at::native::func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }));
  } else {
    at::native::gpu_reduce_kernel<scalar_t, out_t>(
        iter, at::native::func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }));
  }
}

at::Tensor SumDimKernelCuda(
    const at::Tensor& self, at::OptionalIntArrayRef opt_dims,
    bool keepdim, std::optional<at::ScalarType> dtype) {

  auto out_dtype = at::native::get_dtype_from_self(self, dtype, true);
  const bool gpu_lowp_to_f32 =
      (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16) &&
      out_dtype == at::kFloat;
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;

  at::Tensor result = at::empty({0}, self.options().dtype(out_dtype));
  auto iter = at::native::make_reduction(
      "sum", result, self, opt_dims, keepdim, in_dtype, out_dtype);

  if (iter.numel() == 0) {
    result.zero_();
    return result;
  }

  if (iter.dtype() == at::kHalf) {
    SumKernelImpl<at::Half, float>(iter);
  } else if (iter.dtype(1) == at::kHalf && iter.dtype() == at::kFloat) {
    SumKernelImpl<at::Half, float, float>(iter);
  } else if (iter.dtype() == at::kBFloat16) {
    SumKernelImpl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == at::kBFloat16 && iter.dtype() == at::kFloat) {
    SumKernelImpl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "sum_cuda", [&]() {
      SumKernelImpl<scalar_t>(iter);
    });
  }

  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SumDimFn, sum_dim_stub, FlagosDevice::kCuda, SumDimKernelCuda)

} // namespace at::native::flagos
