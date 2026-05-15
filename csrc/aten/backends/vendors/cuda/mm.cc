// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../../mm.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/mm_native.h>
#include <c10/util/Exception.h>

namespace at::native::flagos {

void StructuredMmOutFlagos::set_output_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

void StructuredMmOutFlagos::set_output_raw_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

const at::Tensor& StructuredMmOutFlagos::maybe_get_output(int64_t) {
  return out_;
}

void StructuredMmOutFlagos::impl(const at::Tensor& self, const at::Tensor& mat2, const std::string& op_name) {
  mm_stub.DispatchAs(op_name, self, mat2, out_);
}

namespace {

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

FLAGOS_REGISTER_DISPATCH(MmFn, mm_stub, FlagosDevice::kCuda, MmKernelCuda)

} // namespace at::native::flagos
