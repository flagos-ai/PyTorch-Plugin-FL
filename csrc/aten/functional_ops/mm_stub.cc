// Copyright (c) 2026, BAAI. All rights reserved.

#include "mm_stub.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/mm_native.h>
#include <flag_gems/operators.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(MmFn, mm_stub, "mm")

namespace {

void MmKernelFlaggems(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  flag_gems::mm_out_tensor(self, mat2, out);
}

void MmKernelCuda(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  struct cuda_impl final : public at::native::structured_mm_out_cuda {
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

FLAGOS_REGISTER_DISPATCH(MmFn, mm_stub, FlagosDevice::kFlagOs, MmKernelFlaggems)
FLAGOS_REGISTER_DISPATCH(MmFn, mm_stub, FlagosDevice::kCuda,   MmKernelCuda)

} // namespace at::native::flagos
