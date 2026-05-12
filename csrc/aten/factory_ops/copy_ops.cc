// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "copy_ops.h"

#include <ATen/native/Resize.h>
#include <accelerator/include/flagos.h>

namespace at::native::flagos {

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  at::Tensor self_contig = self.contiguous();
  at::Tensor dst_contig = dst.is_contiguous() ? dst : dst.contiguous();

  size_t nbytes = self_contig.numel() * self_contig.element_size();

  if (self.is_privateuseone() && dst.is_privateuseone()) {
    Memcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
  } else if (self.is_cpu() && dst.is_privateuseone()) {
    Memcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyHostToDevice);
  } else if (self.is_privateuseone() && dst.is_cpu()) {
    Memcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToHost);
  } else if (self.is_privateuseone() && dst.is_cuda()) {
    Memcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
  } else if (self.is_cuda() && dst.is_privateuseone()) {
    Memcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
  } else {
    TORCH_CHECK(false, "Unsupported device combination for copy: ", self.device(), " -> ", dst.device());
  }

  if (!dst.is_contiguous()) {
    dst.copy_(dst_contig);
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::flagos::_copy_from(self, dst, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense expects a tensor with 1 element");
  at::Tensor cpu_tensor = at::empty({1}, self.options().device(at::kCPU));
  Memcpy(cpu_tensor.data_ptr(), self.data_ptr(), self.element_size(), MemcpyDeviceToHost);
  return cpu_tensor.item();
}

at::Tensor _to_copy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format_opt) {

  TORCH_CHECK(
      !layout_opt.has_value() || self.layout() == layout_opt.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout_opt.value());
  TORCH_CHECK(
      self.layout() == c10::kStrided,
      "flagos _to_copy only supports strided tensors, but got ",
      self.layout());
  TORCH_CHECK(
      !self.is_quantized(),
      "flagos _to_copy does not support quantized tensors yet");
  TORCH_CHECK(
      !pin_memory_opt.value_or(false),
      "flagos _to_copy does not support pin_memory=True yet");
  auto device = device_opt.value_or(self.device());
  auto dtype = dtype_opt.value_or(self.scalar_type());
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);

  if ((device.is_privateuseone() || device.is_cuda()) && device.index() < 0) {
    const auto self_device = self.device();
    const auto device_index = self_device.type() == device.type() &&
            self_device.index() >= 0
        ? self_device.index()
        : 0;
    device = c10::Device(device.type(), device_index);
  }
  
  if (device == self.device() && dtype == self.scalar_type()) {
    if (memory_format == c10::MemoryFormat::Preserve) {
      return self.clone();
    }
    return self.clone().contiguous(memory_format);
  }

  bool src_is_flagos = self.is_privateuseone();
  bool dst_is_flagos = device.is_privateuseone();
  bool dst_is_cuda = device.is_cuda();
  bool dst_is_cpu = device.is_cpu();

  at::Tensor result;

  if (src_is_flagos && dst_is_cuda) {
    int device_index = device.index() >= 0 ? device.index() : (self.device().index() >= 0 ? self.device().index() : 0);
    at::Tensor self_contig = self.contiguous();
    at::Tensor temp = at::empty(self_contig.sizes(), self_contig.options().device(c10::Device(c10::kCUDA, device_index)));
    size_t nbytes = self_contig.numel() * self_contig.element_size();
    if (nbytes > 0) {
      Memcpy(temp.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
    }
    result = (dtype != self.scalar_type()) ? temp.to(dtype) : temp;
  } else if (src_is_flagos && dst_is_flagos) {
    int device_index = device.index() >= 0 ? device.index() : 0;
    at::Tensor self_contig = self.contiguous();
    if (dtype != self.scalar_type()) {
      size_t nbytes = self_contig.numel() * self_contig.element_size();
      at::Tensor cpu_tensor = at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
      if (nbytes > 0) {
        Memcpy(cpu_tensor.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToHost);
      }
      cpu_tensor = cpu_tensor.to(dtype);
      result = at::empty(cpu_tensor.sizes(), cpu_tensor.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t result_nbytes = cpu_tensor.numel() * cpu_tensor.element_size();
      if (result_nbytes > 0) {
        Memcpy(result.data_ptr(), cpu_tensor.data_ptr(), result_nbytes, MemcpyHostToDevice);
      }
    } else {
      result = at::empty(self_contig.sizes(), self_contig.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t nbytes = self_contig.numel() * self_contig.element_size();
      if (nbytes > 0) {
        Memcpy(result.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
      }
    }
  } else if (src_is_flagos && dst_is_cpu) {
    at::Tensor self_contig = self.contiguous();
    at::Tensor temp = at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
    size_t nbytes = self_contig.numel() * self_contig.element_size();
    if (nbytes > 0) {
      Memcpy(temp.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToHost);
    }
    result = (dtype != self.scalar_type()) ? temp.to(dtype) : temp;
  } else if (!src_is_flagos && dst_is_flagos) {
    int device_index = device.index() >= 0 ? device.index() : 0;
    at::Tensor src_contig = self.contiguous();
    if (dtype != self.scalar_type()) {
      src_contig = src_contig.to(dtype);
    }
    result = at::empty(src_contig.sizes(), src_contig.options().device(c10::Device(c10::kPrivateUse1, device_index)));
    size_t nbytes = src_contig.numel() * src_contig.element_size();
    if (nbytes > 0) {
      if (self.is_cpu()) {
        Memcpy(result.data_ptr(), src_contig.data_ptr(), nbytes, MemcpyHostToDevice);
      } else if (self.is_cuda()) {
        Memcpy(result.data_ptr(), src_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
      } else {
        TORCH_CHECK(false, "_to_copy: unsupported source device ", self.device());
      }
    }
  } else {
    at::Tensor cpu_tensor = self.to(at::kCPU).to(dtype);
    if (dst_is_flagos) {
      int device_index = device.index() >= 0 ? device.index() : 0;
      result = at::empty(cpu_tensor.sizes(), cpu_tensor.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t nbytes = cpu_tensor.numel() * cpu_tensor.element_size();
      if (nbytes > 0) {
        Memcpy(result.data_ptr(), cpu_tensor.data_ptr(), nbytes, MemcpyHostToDevice);
      }
    } else {
      result = cpu_tensor.to(device);
    }
  }

  if (memory_format != c10::MemoryFormat::Preserve) {
    result = result.contiguous(memory_format);
  }

  return result;
}

} // namespace at::native::flagos
