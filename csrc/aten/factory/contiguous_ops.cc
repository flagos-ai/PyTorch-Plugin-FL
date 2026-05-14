// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contiguous_ops.h"

#include <ATen/native/Resize.h>
#include <accelerator/include/flagos.h>

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

      if (self.dim() == 2) {
        int64_t rows = self.size(0);
        int64_t cols = self.size(1);
        int64_t src_stride0 = self.stride(0);
        int64_t src_stride1 = self.stride(1);
        size_t elem_size = self.element_size();

        char* src_base = static_cast<char*>(storage_cpu.data_ptr()) + self.storage_offset() * elem_size;
        char* dst_base = static_cast<char*>(cpu_contig.data_ptr());

        for (int64_t i = 0; i < rows; i++) {
          for (int64_t j = 0; j < cols; j++) {
            char* src = src_base + (i * src_stride0 + j * src_stride1) * elem_size;
            char* dst = dst_base + (i * cols + j) * elem_size;
            memcpy(dst, src, elem_size);
          }
        }
      } else {
        cpu_contig.copy_(cpu_view);
      }

      size_t nbytes = cpu_contig.numel() * cpu_contig.element_size();
      Memcpy(result.data_ptr(), cpu_contig.data_ptr(), nbytes, MemcpyHostToDevice);
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

  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));
  result.copy_(self);
  return result;
}

} // namespace at::native::flagos
