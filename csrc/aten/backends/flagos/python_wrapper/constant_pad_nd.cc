// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../constant_pad_nd.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>

namespace py = pybind11;

namespace at::native::flagos {

namespace {

at::Tensor ConstantPadNdKernelPython(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {
  std::vector<c10::IValue> args;
  args.emplace_back(self);
  args.emplace_back(pad.vec());
  args.emplace_back(value);
  return CallPythonOp_Generic("constant_pad_nd", args);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(ConstantPadNdFn, constant_pad_nd_dispatcher, Backend::kFlagOsPython, ConstantPadNdKernelPython)

} // namespace at::native::flagos
