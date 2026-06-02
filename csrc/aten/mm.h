// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ops/mm_meta.h>
#include <ATen/ops/mm_native.h>
#include "dispatcher.h"

#include <string>

namespace at::native::flagos {

using MmFn = void (*)(const at::Tensor&, const at::Tensor&, at::Tensor&);
DECLARE_DISPATCHER(MmFn, mm_dispatcher)

// Structured mm op for PrivateUse1 (flagos) dispatch key.
// meta() is inherited from at::meta::structured_mm (PyTorch-generated).
// impl() dispatches to the backend selected by GetBackendForOp().
struct StructuredMmOut final : public at::meta::structured_mm {
  explicit StructuredMmOut(at::Tensor& out) : out_(out) {}

  void set_output_strided(
      int64_t output_idx,
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      at::TensorOptions options,
      at::DimnameList names) override;

  void set_output_raw_strided(
      int64_t output_idx,
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      at::TensorOptions options,
      at::DimnameList names) override;

  const at::Tensor& maybe_get_output(int64_t output_idx) override;

  void impl(const at::Tensor& self, const at::Tensor& mat2, const std::string& op_name);

  at::Tensor& out_;
};

} // namespace at::native::flagos
