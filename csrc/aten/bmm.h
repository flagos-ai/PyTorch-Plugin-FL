// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ops/bmm_meta.h>
#include <ATen/ops/bmm_native.h>
#include "dispatch_stub.h"

#include <string>

namespace at::native::flagos {

using BmmFn = void (*)(const at::Tensor&, const at::Tensor&, at::Tensor&);
FLAGOS_DECLARE_DISPATCH(BmmFn, bmm_stub)

struct StructuredBmmOutFlagos final : public at::meta::structured_bmm {
  explicit StructuredBmmOutFlagos(at::Tensor& out) : out_(out) {}

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
