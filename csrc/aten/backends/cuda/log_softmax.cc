// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../log_softmax.h"

#include <ATen/ops/_log_softmax_meta.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_log_softmax_backward_data_meta.h>
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_backward_data_meta.h>
#include <ATen/ops/_softmax_backward_data_native.h>

#include "../../device_boxing.h"

namespace at::native::flagos {

namespace {

at::Tensor LogSoftmaxKernelCuda(const at::Tensor& self, int64_t dim, bool half_to_float) {
  auto output_dtype = half_to_float ? at::ScalarType::Float : self.scalar_type();
  auto output = at::empty(self.sizes(), self.options().dtype(output_dtype));

  BoxToCuda(self);
  BoxToCuda(output);

  struct CudaImpl final : public at::native::structured_log_softmax_cuda_out {
    CudaImpl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(int64_t, at::IntArrayRef, at::IntArrayRef,
                                at::TensorOptions, at::DimnameList) override {}
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  CudaImpl op(output);
  op.meta(self, dim, half_to_float);
  op.impl(self, dim, half_to_float, output);

  UnboxToFlagos(self);
  UnboxToFlagos(output);
  return output;
}

at::Tensor LogSoftmaxBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType input_dtype) {
  auto grad_input = at::empty(grad_output.sizes(), grad_output.options());

  BoxToCuda(grad_output);
  BoxToCuda(output);
  BoxToCuda(grad_input);

  struct CudaImpl final : public at::native::structured_log_softmax_backward_cuda_out {
    CudaImpl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(int64_t, at::IntArrayRef, at::IntArrayRef,
                                at::TensorOptions, at::DimnameList) override {}
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  CudaImpl op(grad_input);
  op.meta(grad_output, output, dim, input_dtype);
  op.impl(grad_output, output, dim, input_dtype, grad_input);

  UnboxToFlagos(grad_output);
  UnboxToFlagos(output);
  UnboxToFlagos(grad_input);
  return grad_input;
}

at::Tensor SoftmaxBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType input_dtype) {
  auto grad_input = at::empty(grad_output.sizes(), grad_output.options());

  BoxToCuda(grad_output);
  BoxToCuda(output);
  BoxToCuda(grad_input);

  struct CudaImpl final : public at::native::structured_softmax_backward_cuda_out {
    CudaImpl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(int64_t, at::IntArrayRef, at::IntArrayRef,
                                at::TensorOptions, at::DimnameList) override {}
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  CudaImpl op(grad_input);
  op.meta(grad_output, output, dim, input_dtype);
  op.impl(grad_output, output, dim, input_dtype, grad_input);

  UnboxToFlagos(grad_output);
  UnboxToFlagos(output);
  UnboxToFlagos(grad_input);
  return grad_input;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(LogSoftmaxFn, log_softmax_dispatcher, Backend::kCuda, LogSoftmaxKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(LogSoftmaxBackwardFn, log_softmax_backward_dispatcher, Backend::kCuda, LogSoftmaxBackwardKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(SoftmaxBackwardFn, softmax_backward_dispatcher, Backend::kCuda, SoftmaxBackwardKernelCuda)

} // namespace at::native::flagos
