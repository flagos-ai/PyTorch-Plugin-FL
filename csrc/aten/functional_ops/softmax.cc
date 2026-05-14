// Copyright (c) 2026, BAAI. All rights reserved.

#include "softmax.h"

#include <ATen/ops/_softmax_meta.h>
#include <ATen/ops/_softmax_native.h>
#include <flag_gems/operators.h>

#include "../device_boxing.h"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(SoftmaxFn, softmax_stub, "_softmax")

namespace {

at::Tensor SoftmaxKernelFlaggems(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return flag_gems::softmax(self, dim, half_to_float);
}

at::Tensor SoftmaxKernelCuda(const at::Tensor& self, int64_t dim, bool half_to_float) {
  // Determine output dtype/shape via meta
  auto output_dtype = half_to_float ? at::ScalarType::Float : self.scalar_type();
  auto output = at::empty(self.sizes(), self.options().dtype(output_dtype));

  // Box both to CUDA for the structured kernel
  BoxToCuda(self);
  BoxToCuda(output);

  struct CudaImpl final : public at::native::structured_softmax_cuda_out {
    CudaImpl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(int64_t, at::IntArrayRef, at::IntArrayRef,
                                at::TensorOptions, at::DimnameList) override {
      // Output already allocated, nothing to do
    }
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  CudaImpl op(output);
  op.meta(self, dim, half_to_float);
  op.impl(self, dim, half_to_float, output);

  // Unbox back to flagos
  UnboxToFlagos(self);
  UnboxToFlagos(output);
  return output;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SoftmaxFn, softmax_stub, FlagosDevice::kFlagOs, SoftmaxKernelFlaggems)
FLAGOS_REGISTER_DISPATCH(SoftmaxFn, softmax_stub, FlagosDevice::kCuda, SoftmaxKernelCuda)

} // namespace at::native::flagos
