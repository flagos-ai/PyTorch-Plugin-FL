// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../nll_loss.h"

#include <include/flagos.h>

#include <c10/core/DispatchKey.h>

#include <ATen/ops/nll_loss_backward.h>
#include <ATen/ops/nll_loss_forward.h>

namespace at::native::flagos {

namespace {

bool TensorIsCpuOnly(const at::Tensor& tensor) {
  const c10::DispatchKeySet keys = tensor.key_set();
  return keys.has(c10::DispatchKey::CPU) &&
      !keys.has(c10::DispatchKey::PrivateUse1);
}

int64_t GetDeviceIndex(const at::Tensor& ref, int64_t fallback = 0) {
  if (!ref.key_set().has(c10::DispatchKey::PrivateUse1)) {
    return fallback;
  }
  const c10::Device device = ref.device();
  return device.index() >= 0 ? device.index() : fallback;
}

at::Tensor CopyToCpu(const at::Tensor& tensor, c10::ScalarType dtype) {
  // Autograd saved tensors may not implement Tensor::device() from C++ even when
  // Python reports flagos:0. Use dispatch keys + direct Memcpy instead of .cpu().
  if (TensorIsCpuOnly(tensor)) {
    return tensor.contiguous();
  }
  const at::Tensor src = tensor.contiguous();
  at::Tensor dst = at::empty(
      src.sizes(), at::TensorOptions().dtype(dtype).device(at::kCPU));
  const size_t nbytes = static_cast<size_t>(src.numel()) *
      static_cast<size_t>(c10::elementSize(dtype));
  if (nbytes > 0) {
    Memcpy(dst.data_ptr(), src.data_ptr(), nbytes, MemcpyDeviceToHost);
  }
  return dst;
}

at::Tensor CopyToDevice(
    const at::Tensor& cpu_tensor,
    const at::Tensor& device_ref) {
  const at::Tensor src = cpu_tensor.contiguous();
  const int64_t device_index = GetDeviceIndex(device_ref, 0);
  at::Tensor dst = at::empty(
      src.sizes(),
      at::TensorOptions().dtype(src.scalar_type()).device(
          c10::Device(c10::kPrivateUse1, device_index)));
  const size_t nbytes =
      static_cast<size_t>(src.numel()) * static_cast<size_t>(src.element_size());
  if (nbytes > 0) {
    Memcpy(dst.data_ptr(), src.data_ptr(), nbytes, MemcpyHostToDevice);
  }
  return dst;
}

std::tuple<at::Tensor, at::Tensor> NllLossForwardKernelMetax(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  const at::Tensor self_cpu = CopyToCpu(self, self.scalar_type());
  const at::Tensor target_cpu = CopyToCpu(target, target.scalar_type());
  std::optional<at::Tensor> weight_cpu = std::nullopt;
  if (weight.has_value() && weight->defined()) {
    weight_cpu = CopyToCpu(*weight, weight->scalar_type());
  }
  auto [out_cpu, total_weight_cpu] =
      at::nll_loss_forward(self_cpu, target_cpu, weight_cpu, reduction, ignore_index);
  return std::make_tuple(
      CopyToDevice(out_cpu, self),
      CopyToDevice(total_weight_cpu, self));
}

at::Tensor NllLossBackwardKernelMetax(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  // Saved autograd tensors may not expose a valid scalar_type() from C++.
  const at::Tensor grad_output_cpu = CopyToCpu(grad_output, at::kFloat);
  const at::Tensor self_cpu = CopyToCpu(self, at::kFloat);
  const at::Tensor target_cpu = CopyToCpu(target, at::kLong);
  std::optional<at::Tensor> weight_cpu = std::nullopt;
  if (weight.has_value() && weight->defined()) {
    weight_cpu = CopyToCpu(*weight, at::kFloat);
  }
  const at::Tensor total_weight_cpu = CopyToCpu(total_weight, at::kFloat);
  at::Tensor out_cpu = at::nll_loss_backward(
      grad_output_cpu,
      self_cpu,
      target_cpu,
      weight_cpu,
      reduction,
      ignore_index,
      total_weight_cpu);
  return CopyToDevice(out_cpu, grad_output);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    NllLossForwardFn,
    nll_loss_forward_dispatcher,
    Backend::kMetax,
    NllLossForwardKernelMetax)
REGISTER_IMPL_TO_DISPATCHER(
    NllLossBackwardFn,
    nll_loss_backward_dispatcher,
    Backend::kMetax,
    NllLossBackwardKernelMetax)

} // namespace at::native::flagos
