// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../../nll_loss.h"

#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/nll_loss_backward_native.h>

namespace at::native::flagos {

namespace {

std::tuple<at::Tensor, at::Tensor> NllLossForwardKernelCuda(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
  at::Tensor output, total_weight;
  struct CudaImpl final : public at::native::structured_nll_loss_forward_out_cuda {
    at::Tensor output_, total_weight_;
    void set_output_raw_strided(int64_t idx, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options, at::DimnameList) override {
      if (strides.empty()) {
        if (idx == 0) {
          output_ = at::empty(sizes, options);
        } else {
          total_weight_ = at::empty(sizes, options);
        }
      } else {
        if (idx == 0) {
          output_ = at::empty_strided(sizes, strides, options);
        } else {
          total_weight_ = at::empty_strided(sizes, strides, options);
        }
      }
    }
    const at::Tensor& maybe_get_output(int64_t idx) override {
      return idx == 0 ? output_ : total_weight_;
    }
  };
  CudaImpl op;
  at::OptionalTensorRef weight_ref = weight.has_value()
      ? at::OptionalTensorRef(*weight) : at::OptionalTensorRef();
  op.meta(self, target, weight_ref, reduction, ignore_index);
  op.impl(self, target, weight_ref, reduction, ignore_index, op.output_, op.total_weight_);
  return std::make_tuple(op.output_, op.total_weight_);
}

at::Tensor NllLossBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) {
  at::Tensor grad_input;
  struct CudaImpl final : public at::native::structured_nll_loss_backward_out_cuda {
    at::Tensor grad_input_;
    void set_output_raw_strided(int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options, at::DimnameList) override {
      if (strides.empty()) {
        grad_input_ = at::empty(sizes, options);
      } else {
        grad_input_ = at::empty_strided(sizes, strides, options);
      }
    }
    const at::Tensor& maybe_get_output(int64_t) override { return grad_input_; }
  };
  CudaImpl op;
  at::OptionalTensorRef weight_ref = weight.has_value()
      ? at::OptionalTensorRef(*weight) : at::OptionalTensorRef();
  op.meta(grad_output, self, target, weight_ref, reduction, ignore_index, total_weight);
  op.impl(grad_output, self, target, weight_ref, reduction, ignore_index, total_weight, op.grad_input_);
  return op.grad_input_;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(NllLossForwardFn, nll_loss_forward_stub, FlagosDevice::kCuda, NllLossForwardKernelCuda)
FLAGOS_REGISTER_DISPATCH(NllLossBackwardFn, nll_loss_backward_stub, FlagosDevice::kCuda, NllLossBackwardKernelCuda)

} // namespace at::native::flagos
