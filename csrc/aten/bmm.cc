// Copyright (c) 2026, BAAI. All rights reserved.

#include "bmm.h"

namespace at::native::flagos {

void StructuredBmmOut::set_output_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  out_.resize_(sizes);
}

void StructuredBmmOut::set_output_raw_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  out_.resize_(sizes);
}

const at::Tensor& StructuredBmmOut::maybe_get_output(int64_t) {
  return out_;
}

void StructuredBmmOut::impl(const at::Tensor& self, const at::Tensor& mat2, const std::string& op_name) {
  bmm_dispatcher.DispatchAs(op_name, self, mat2, out_);
}

ADD_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, "bmm")

} // namespace at::native::flagos
