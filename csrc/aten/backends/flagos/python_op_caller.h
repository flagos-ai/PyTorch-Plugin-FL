// Copyright (c) 2026, BAAI. All rights reserved.
//
// Python op caller: bridge from C++ stub layer to FlagGems Python wrapper ops
// via pybind11 embedded interpreter.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>
#include <c10/util/ArrayRef.h>

namespace at::native::flagos {

// Call a FlagGems Python op by function name.
// All variants acquire the GIL internally and look up the function
// from the cached flag_gems.ops module.

// (Tensor) -> Tensor
at::Tensor CallPythonOp_T(const char* func_name, const at::Tensor& self);

// (Tensor) -> Tensor  [inplace: returns self]
at::Tensor& CallPythonOp_T_inplace(const char* func_name, at::Tensor& self);

// (Tensor, Tensor) -> Tensor
at::Tensor CallPythonOp_TT(const char* func_name, const at::Tensor& a, const at::Tensor& b);

// (Tensor, Tensor, Scalar) -> Tensor
at::Tensor CallPythonOp_TTS(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha);

// (Tensor, Scalar) -> Tensor
at::Tensor CallPythonOp_TS(const char* func_name, const at::Tensor& self, const at::Scalar& other);

// (Tensor, int64_t, bool) -> Tensor
at::Tensor CallPythonOp_TIB(const char* func_name, const at::Tensor& self, int64_t dim, bool flag);

// (Tensor, OptionalIntArrayRef, bool, optional<ScalarType>) -> Tensor
at::Tensor CallPythonOp_TOIB(const char* func_name, const at::Tensor& self,
                              at::OptionalIntArrayRef dim, bool keepdim,
                              std::optional<at::ScalarType> dtype);

// (Tensor, Tensor, Tensor) -> Tensor
at::Tensor CallPythonOp_TTT(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

// (Tensor, optional<ScalarType>) -> Tensor  [keyword: dtype=...]
at::Tensor CallPythonOp_TD(const char* func_name, const at::Tensor& self,
                            std::optional<at::ScalarType> dtype);

// Generic variadic caller using Python *args/**kwargs.
// For ops with complex signatures not covered above.
// Arguments are passed as a vector of IValues.
at::Tensor CallPythonOp_Generic(const char* func_name, const std::vector<c10::IValue>& args);

} // namespace at::native::flagos
