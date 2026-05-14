// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "../common.h"

namespace at::native::flagos {

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset);

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format);

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride);

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size);

} // namespace at::native::flagos
