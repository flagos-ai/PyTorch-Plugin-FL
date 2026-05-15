// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mean.h"

#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <ATen/native/cuda/Reduce.cuh>

namespace at::native::flagos {

namespace {

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void MeanKernelImpl(at::TensorIterator& iter) {
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor = static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  constexpr bool is_16_bits = sizeof(scalar_t) == 2;
  if constexpr (is_16_bits) {
    at::native::gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, at::native::MeanOps<scalar_t, acc_t, factor_t, out_t>{factor});
  } else {
    at::native::gpu_reduce_kernel<scalar_t, out_t>(
        iter, at::native::MeanOps<scalar_t, acc_t, factor_t, out_t>{factor});
  }
}

at::Tensor MeanDimKernelCuda(
    const at::Tensor& self, at::OptionalIntArrayRef opt_dims,
    bool keepdim, std::optional<at::ScalarType> dtype) {

  auto out_dtype = at::native::get_dtype_from_self(self, dtype, true);
  const bool gpu_lowp_to_f32 =
      (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16) &&
      out_dtype == at::kFloat;
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;

  at::Tensor result = at::empty({0}, self.options().dtype(out_dtype));
  auto iter = at::native::make_reduction(
      "mean", result, self, opt_dims, keepdim, in_dtype, out_dtype);

  if (iter.numel() == 0) {
    return result;
  }

  if (iter.dtype() == at::kHalf) {
    MeanKernelImpl<at::Half, float>(iter);
  } else if (iter.dtype(1) == at::kHalf && iter.dtype() == at::kFloat) {
    MeanKernelImpl<at::Half, float, float>(iter);
  } else if (iter.dtype() == at::kBFloat16) {
    MeanKernelImpl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == at::kBFloat16 && iter.dtype() == at::kFloat) {
    MeanKernelImpl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cuda", [&]() {
      MeanKernelImpl<scalar_t>(iter);
    });
  }

  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MeanDimFn, mean_dim_stub, FlagosDevice::kCuda, MeanDimKernelCuda)

} // namespace at::native::flagos
