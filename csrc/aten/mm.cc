// Copyright (c) 2026, BAAI. All rights reserved.

#include "mm.h"

namespace at::native::flagos {

void StructuredMmOut::set_output_strided(
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

void StructuredMmOut::set_output_raw_strided(
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

const at::Tensor& StructuredMmOut::maybe_get_output(int64_t) {
  return out_;
}

void StructuredMmOut::impl(const at::Tensor& self, const at::Tensor& mat2, const std::string& op_name) {
  mm_dispatcher.DispatchAs(op_name, self, mat2, out_);
}

ADD_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, "mm")

} // namespace at::native::flagos
