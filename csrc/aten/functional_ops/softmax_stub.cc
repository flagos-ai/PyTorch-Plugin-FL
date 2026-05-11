// Copyright (c) 2026, BAAI. All rights reserved.

#include "softmax_stub.h"

#include <ATen/ops/_softmax_meta.h>
#include <ATen/ops/_softmax_native.h>
#include <flag_gems/operators.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(SoftmaxFn, softmax_stub, "_softmax")

namespace {

at::Tensor SoftmaxKernelFlaggems(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return flag_gems::softmax(self, dim, half_to_float);
}

at::Tensor SoftmaxKernelCuda(const at::Tensor& self, int64_t dim, bool half_to_float) {
  at::Tensor out;
  struct CudaImpl final : public at::native::structured_softmax_cuda_out {
    CudaImpl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options, at::DimnameList) override {
      if (strides.empty()) {
        out_ = at::empty(sizes, options);
      } else {
        out_ = at::empty_strided(sizes, strides, options);
      }
    }
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  CudaImpl op(out);
  op.meta(self, dim, half_to_float);
  op.impl(self, dim, half_to_float, out);
  return out;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SoftmaxFn, softmax_stub, FlagosDevice::kFlagOs, SoftmaxKernelFlaggems)
FLAGOS_REGISTER_DISPATCH(SoftmaxFn, softmax_stub, FlagosDevice::kCuda, SoftmaxKernelCuda)

} // namespace at::native::flagos
