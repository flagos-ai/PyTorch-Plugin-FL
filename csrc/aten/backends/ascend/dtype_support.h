// Copyright (c) 2026, BAAI. All rights reserved.
//
// Dtype capability helpers for Ascend aclnn kernels.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <c10/core/ScalarType.h>
#include <string_view>

namespace at::native::flagos::ascend {

// aclnn matmul accepts floating-point inputs up to float32 on the supported
// CANN release. Storage and elementwise kernels can still support float64, so
// matmul callers must use the CPU fallback to preserve PyTorch semantics.
inline bool IsMatmulDtypeSupported(c10::ScalarType dtype) {
  return dtype == c10::kHalf || dtype == c10::kBFloat16 || dtype == c10::kFloat;
}

// CANN's generated unary kernels have inconsistent integral contracts: some
// reject an integral dtype, some preserve it when PyTorch promotes to float,
// and others accept only a subset of integer widths. Use native ACLNN only for
// real floating-point inputs and preserve PyTorch semantics through the CPU
// fallback for every integral and bool input.
inline bool IsUnaryDtypeSupported(c10::ScalarType dtype) {
  return dtype == c10::kHalf || dtype == c10::kBFloat16 ||
      dtype == c10::kFloat || dtype == c10::kDouble;
}

inline c10::ScalarType BinaryResultType(
    std::string_view op,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto dtype = at::result_type(self, other);
  if (op == "aclnnDiv" && c10::isIntegralType(dtype, true)) {
    return at::get_default_dtype_as_scalartype();
  }
  return dtype;
}

// CPU fallback for an Ascend operation whose native aclnn implementation does
// not accept the requested dtype. The caller supplies the already-dispatched
// CPU result and this helper only moves it back to the original device.
inline at::Tensor MoveCpuResultToDevice(
    at::Tensor cpu_result,
    const at::Tensor& reference) {
  return std::move(cpu_result).to(reference.device());
}

} // namespace at::native::flagos::ascend
