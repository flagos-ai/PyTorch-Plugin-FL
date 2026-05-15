// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "set_ops.h"

#include <ATen/native/Resize.h>

namespace at::native::flagos {

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  // Directly manipulate TensorImpl to avoid infinite recursion.
  // at::native::set_ internally calls self.set_(source, 0, ...) which
  // dispatches back to set_.source_Storage_storage_offset, and that
  // implementation used to call result.set_(storage) which dispatches
  // back here, causing infinite recursion and stack overflow.
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  int64_t stride_val = 1;
  self.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(source));
  self.unsafeGetTensorImpl()->set_storage_offset(0);
  self.unsafeGetTensorImpl()->set_sizes_and_strides(
      c10::IntArrayRef(&new_size, 1), c10::IntArrayRef(&stride_val, 1));
  return self;
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  // Directly manipulate TensorImpl to avoid infinite recursion.
  // Previously called result.set_(storage) which dispatches through the
  // dispatcher back to set_.source_Storage, creating an infinite loop.
  result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return result;
}

} // namespace at::native::flagos
