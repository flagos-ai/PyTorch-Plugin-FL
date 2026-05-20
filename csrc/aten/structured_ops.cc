// Copyright (c) 2026, BAAI. All rights reserved.
// Shared structured-op glue for mm/bmm (used when CUDA/FlagGems backends are filtered out).

#include "bmm.h"
#include "mm.h"

#include <ATen/native/Resize.h>

namespace at::native::flagos {

void StructuredMmOutFlagos::set_output_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

void StructuredMmOutFlagos::set_output_raw_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

const at::Tensor& StructuredMmOutFlagos::maybe_get_output(int64_t) {
  return out_;
}

void StructuredMmOutFlagos::impl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    const std::string& op_name) {
  mm_stub.DispatchAs(op_name, self, mat2, out_);
}

void StructuredBmmOutFlagos::set_output_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

void StructuredBmmOutFlagos::set_output_raw_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

const at::Tensor& StructuredBmmOutFlagos::maybe_get_output(int64_t) {
  return out_;
}

void StructuredBmmOutFlagos::impl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    const std::string& op_name) {
  bmm_stub.DispatchAs(op_name, self, mat2, out_);
}

} // namespace at::native::flagos
