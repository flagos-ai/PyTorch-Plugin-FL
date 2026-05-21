// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../nll_loss.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

std::tuple<at::Tensor, at::Tensor> NllLossForwardKernelAscend(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index) {
  namespace ascend = at::native::flagos::ascend;

  at::Tensor output;
  if (reduction == 0) {  // None
    output = ascend::OpPreparation::apply_tensor_without_format(
        {self.size(0)}, self.options());
  } else {
    output = ascend::OpPreparation::apply_tensor_without_format(
        {}, self.options());
  }
  auto total_weight = ascend::OpPreparation::apply_tensor_without_format(
      {}, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_output(output);
  ascend::AclTensorWrapper acl_total_weight(total_weight);

  if (weight.has_value()) {
    ascend::AclTensorWrapper acl_weight(weight.value());
    EXEC_ASCEND_CMD(aclnnNllLoss, acl_self.get(), acl_target.get(),
                    acl_weight.get(), reduction, ignore_index,
                    acl_output.get(), acl_total_weight.get());
  } else {
    EXEC_ASCEND_CMD(aclnnNllLoss, acl_self.get(), acl_target.get(),
                    nullptr, reduction, ignore_index,
                    acl_output.get(), acl_total_weight.get());
  }
  return std::make_tuple(output, total_weight);
}

at::Tensor NllLossBackwardKernelAscend(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_total_weight(total_weight);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  if (weight.has_value()) {
    ascend::AclTensorWrapper acl_weight(weight.value());
    EXEC_ASCEND_CMD(aclnnNllLossBackward, acl_grad_output.get(), acl_self.get(),
                    acl_target.get(), acl_weight.get(), reduction, ignore_index,
                    acl_total_weight.get(), acl_grad_input.get());
  } else {
    EXEC_ASCEND_CMD(aclnnNllLossBackward, acl_grad_output.get(), acl_self.get(),
                    acl_target.get(), nullptr, reduction, ignore_index,
                    acl_total_weight.get(), acl_grad_input.get());
  }
  return grad_input;
}

FLAGOS_REGISTER_DISPATCH(NllLossForwardFn, nll_loss_forward_stub, FlagosDevice::kAscend, NllLossForwardKernelAscend)
FLAGOS_REGISTER_DISPATCH(NllLossBackwardFn, nll_loss_backward_stub, FlagosDevice::kAscend, NllLossBackwardKernelAscend)

} // namespace at::native::flagos
