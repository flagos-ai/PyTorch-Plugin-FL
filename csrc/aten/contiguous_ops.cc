// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contiguous_ops.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/copy_native.h>
#include <include/flagos.h>
#include "device_boxing.h"

namespace at::native::flagos {

at::Tensor contiguous(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));

  if (self.is_privateuseone()) {
    int64_t numel = self.numel();
    if (numel > 0) {
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO)
      // CUDA platform: use DeviceBoxingGuard to invoke native CUDA strided copy
      // kernel on-device, avoiding expensive CPU round-trip.
      DeviceBoxingGuard guard(self, result);
      at::native::copy_(result, self, false);
#else
      // Ascend: no CUDA runtime, fall back to CPU round-trip.
      size_t storage_size = self.storage().nbytes();
      at::Tensor storage_cpu = at::empty(
          {static_cast<int64_t>(storage_size)},
          at::TensorOptions().dtype(at::kByte).device(at::kCPU));
      Memcpy(storage_cpu.data_ptr(), self.storage().data(), storage_size, MemcpyDeviceToHost);

      at::Tensor cpu_view = at::empty({0}, self.options().device(at::kCPU));
      cpu_view.set_(
          storage_cpu.storage(),
          self.storage_offset(),
          self.sizes(),
          self.strides());

      auto cpu_contig = at::empty(self.sizes(), self.options().device(at::kCPU).memory_format(memory_format));
      cpu_contig.copy_(cpu_view);

      size_t nbytes = cpu_contig.numel() * cpu_contig.element_size();
      Memcpy(result.data_ptr(), cpu_contig.data_ptr(), nbytes, MemcpyHostToDevice);
#endif
    }

    return result;
  }

  result.copy_(self);
  return result;
}

at::Tensor clone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);

  if (memory_format == c10::MemoryFormat::Preserve) {
    if (self.is_contiguous()) {
      auto result = at::empty_like(self);
      size_t nbytes = self.numel() * self.element_size();
      if (nbytes > 0 && self.is_privateuseone()) {
        Memcpy(result.data_ptr(), self.data_ptr(), nbytes, MemcpyDeviceToDevice);
      } else if (nbytes > 0) {
        result.copy_(self);
      }
      return result;
    }
    // For non-contiguous with Preserve, fall through to create contiguous copy.
    // NOTE: Do NOT call contiguous() here to avoid infinite recursion when
    // contiguous is unregistered from PrivateUse1 (CompositeImplicitAutograd
    // contiguous calls clone, which would call contiguous again).
    memory_format = c10::MemoryFormat::Contiguous;
  }

  // Non-contiguous clone: use DeviceBoxingGuard to leverage CUDA's native
  // strided copy kernel instead of expensive CPU round-trip.
  if (self.is_privateuseone()) {
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO)
    auto result = at::empty(
        self.sizes(), self.options().memory_format(memory_format));
    DeviceBoxingGuard guard(self, result);
    at::native::copy_(result, self, false);
    return result;
#else
    auto result = at::empty(
        self.sizes(), self.options().memory_format(memory_format));
    result.copy_(self);
    return result;
#endif
  }

  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));
  result.copy_(self);
  return result;
}

} // namespace at::native::flagos
