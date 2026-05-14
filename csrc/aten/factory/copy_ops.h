// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "../common.h"

namespace at::native::flagos {

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking);

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst);

at::Scalar _local_scalar_dense(const at::Tensor& self);

at::Tensor _to_copy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format);

} // namespace at::native::flagos
