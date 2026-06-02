// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cat.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/cat_native.h>

namespace at::native::flagos {

namespace {

at::Tensor CatKernelCuda(const at::ITensorListRef& tensors, int64_t dim) {
  // Call the CUDA structured kernel directly, bypassing at::cuda::cat()
  // which triggers CUDAGuardImpl device-type check.
  // flagos and CUDA share the same GPU memory so this is safe.
  struct cuda_impl final : public at::native::structured_cat_out_cuda {
    at::Tensor out_;
    void set_output_strided(
        int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
        at::TensorOptions options, at::DimnameList) override {
      out_ = at::empty(sizes, options);
    }
    void set_output_raw_strided(
        int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
        at::TensorOptions options, at::DimnameList) override {
      out_ = at::empty(sizes, options);
    }
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
  };

  cuda_impl op;
  auto precompute = op.meta(tensors, dim);
  op.impl(tensors, precompute.dim, precompute.valid,
          precompute.all_contiguous, precompute.all_same_dtype,
          precompute.all_same_sizes_and_stride, precompute.memory_format,
          op.out_);
  return op.out_;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kCuda, CatKernelCuda)

} // namespace at::native::flagos
