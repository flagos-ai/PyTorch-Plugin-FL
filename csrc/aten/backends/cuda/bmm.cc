// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bmm.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/bmm_native.h>
#include <c10/util/Exception.h>

namespace at::native::flagos {

namespace {

void BmmKernelCuda(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  struct cuda_impl final : public at::native::structured_bmm_out_cuda {
    explicit cuda_impl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(
        int64_t, at::IntArrayRef sizes, at::IntArrayRef,
        at::TensorOptions, at::DimnameList) override {
      at::native::resize_output(out_, sizes);
    }
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  cuda_impl op(out);
  op.impl(self, mat2, out);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kCuda, BmmKernelCuda)

} // namespace at::native::flagos
