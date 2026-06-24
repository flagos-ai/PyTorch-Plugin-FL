// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../nll_loss.h"

#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/nll_loss_backward_native.h>

#include "../../device_boxing.h"

namespace at::native::flagos {

namespace {

std::tuple<at::Tensor, at::Tensor> NllLossForwardKernelCuda(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
  // Box inputs to CUDA
  BoxToCuda(self);
  BoxToCuda(target);
  if (weight.has_value() && weight->defined()) {
    BoxToCuda(*weight);
  }

  struct CudaImpl final : public at::native::structured_nll_loss_forward_out_cuda {
    at::Tensor output_, total_weight_;
    void set_output_raw_strided(int64_t idx, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options, at::DimnameList) override {
      // Force CUDA device for output allocation
      auto cuda_options = options.device(c10::DeviceType::CUDA);
      if (strides.empty()) {
        if (idx == 0) {
          output_ = at::empty(sizes, cuda_options);
        } else {
          total_weight_ = at::empty(sizes, cuda_options);
        }
      } else {
        if (idx == 0) {
          output_ = at::empty_strided(sizes, strides, cuda_options);
        } else {
          total_weight_ = at::empty_strided(sizes, strides, cuda_options);
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

  // Unbox inputs
  UnboxToFlagos(self);
  UnboxToFlagos(target);
  if (weight.has_value() && weight->defined()) {
    UnboxToFlagos(*weight);
  }
  // Unbox outputs
  UnboxToFlagos(op.output_);
  UnboxToFlagos(op.total_weight_);
  return std::make_tuple(op.output_, op.total_weight_);
}

at::Tensor NllLossBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) {
  // Box inputs to CUDA
  BoxToCuda(grad_output);
  BoxToCuda(self);
  BoxToCuda(target);
  BoxToCuda(total_weight);
  if (weight.has_value() && weight->defined()) {
    BoxToCuda(*weight);
  }

  struct CudaImpl final : public at::native::structured_nll_loss_backward_out_cuda {
    at::Tensor grad_input_;
    void set_output_raw_strided(int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options, at::DimnameList) override {
      auto cuda_options = options.device(c10::DeviceType::CUDA);
      if (strides.empty()) {
        grad_input_ = at::empty(sizes, cuda_options);
      } else {
        grad_input_ = at::empty_strided(sizes, strides, cuda_options);
      }
    }
    const at::Tensor& maybe_get_output(int64_t) override { return grad_input_; }
  };
  CudaImpl op;
  at::OptionalTensorRef weight_ref = weight.has_value()
      ? at::OptionalTensorRef(*weight) : at::OptionalTensorRef();
  op.meta(grad_output, self, target, weight_ref, reduction, ignore_index, total_weight);
  op.impl(grad_output, self, target, weight_ref, reduction, ignore_index, total_weight, op.grad_input_);

  // Unbox inputs
  UnboxToFlagos(grad_output);
  UnboxToFlagos(self);
  UnboxToFlagos(target);
  UnboxToFlagos(total_weight);
  if (weight.has_value() && weight->defined()) {
    UnboxToFlagos(*weight);
  }
  // Unbox output
  UnboxToFlagos(op.grad_input_);
  return op.grad_input_;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(NllLossForwardFn, nll_loss_forward_dispatcher, Backend::kCuda, NllLossForwardKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(NllLossBackwardFn, nll_loss_backward_dispatcher, Backend::kCuda, NllLossBackwardKernelCuda)

} // namespace at::native::flagos
