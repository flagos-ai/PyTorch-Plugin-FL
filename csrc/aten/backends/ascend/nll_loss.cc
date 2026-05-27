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
        {1}, self.options());
  }
  auto total_weight = ascend::OpPreparation::apply_tensor_without_format(
      {1}, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_output(output);
  ascend::AclTensorWrapper acl_total_weight(total_weight);

  // ACL does not accept nullptr for weight; create a dummy ones tensor if not provided
  at::Tensor weight_tensor;
  if (weight.has_value() && weight.value().numel() > 0) {
    weight_tensor = weight.value();
  } else {
    weight_tensor = at::ones({self.size(1)}, self.options().device(at::kCPU)).to(self.device());
    aclrtSynchronizeStream(ascend::GetCurrentAclStream());
  }
  ascend::AclTensorWrapper acl_weight(weight_tensor);

  EXEC_ASCEND_CMD(aclnnNLLLoss, acl_self.get(), acl_target.get(),
                  acl_weight.get(), reduction, ignore_index,
                  acl_output.get(), acl_total_weight.get());
  aclrtSynchronizeStream(ascend::GetCurrentAclStream());

  if (reduction != 0) {
    output = output.squeeze(0);
    total_weight = total_weight.squeeze(0);
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

  // ACL expects gradOutput and totalWeight to have shape (1,) not scalar
  auto grad_out_1d = grad_output.dim() == 0 ? grad_output.unsqueeze(0) : grad_output;
  auto total_weight_1d = total_weight.dim() == 0 ? total_weight.unsqueeze(0) : total_weight;

  ascend::AclTensorWrapper acl_grad_output(grad_out_1d);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_total_weight(total_weight_1d);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  // ACL does not accept nullptr for weight; create a dummy ones tensor if not provided
  at::Tensor weight_tensor;
  if (weight.has_value() && weight.value().numel() > 0) {
    weight_tensor = weight.value();
  } else {
    weight_tensor = at::ones({self.size(1)}, self.options().device(at::kCPU)).to(self.device());
    aclrtSynchronizeStream(ascend::GetCurrentAclStream());
  }
  ascend::AclTensorWrapper acl_weight(weight_tensor);

  EXEC_ASCEND_CMD(aclnnNLLLossBackward, acl_grad_output.get(), acl_self.get(),
                  acl_target.get(), acl_weight.get(), reduction, ignore_index,
                  acl_total_weight.get(), acl_grad_input.get());
  aclrtSynchronizeStream(ascend::GetCurrentAclStream());
  return grad_input;
}

FLAGOS_REGISTER_DISPATCH(NllLossForwardFn, nll_loss_forward_stub, FlagosDevice::kAscend, NllLossForwardKernelAscend)
FLAGOS_REGISTER_DISPATCH(NllLossBackwardFn, nll_loss_backward_stub, FlagosDevice::kAscend, NllLossBackwardKernelAscend)

} // namespace at::native::flagos
