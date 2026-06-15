// Copyright (c) 2026, BAAI. All rights reserved.
//
// CUDA backend dispatch registration for:
// - aten::_local_scalar_dense.default
// - aten::_to_copy.default

#include "../../copy_dispatcher.h"

#include "../../copy_ops.h"

namespace at::native::flagos {

namespace {

at::Scalar LocalScalarDenseKernelCuda(const at::Tensor& self) {
  return ::at::native::flagos::_local_scalar_dense(self);
}

at::Tensor ToCopyKernelCuda(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format) {
  return ::at::native::flagos::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

}  // namespace

REGISTER_IMPL_TO_DISPATCHER(
    LocalScalarDenseFn,
    local_scalar_dense_dispatcher,
    Backend::kCuda,
    LocalScalarDenseKernelCuda);
REGISTER_IMPL_TO_DISPATCHER(
    ToCopyFn, to_copy_dispatcher, Backend::kCuda, ToCopyKernelCuda);

}  // namespace at::native::flagos
