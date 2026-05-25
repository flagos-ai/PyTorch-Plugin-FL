// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../zeros.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>

namespace py = pybind11;

namespace at::native::flagos {

namespace {

at::Tensor PythonToTensorLocal(const py::object& obj) {
  if (obj.is_none()) return at::Tensor();
  PyObject* raw = obj.ptr();
  TORCH_CHECK(THPVariable_Check(raw), "Expected a Tensor from Python op");
  return THPVariable_Unpack(raw);
}

at::Tensor ZerosKernelPython(at::IntArrayRef size,
                             std::optional<at::ScalarType> dtype,
                             std::optional<at::Layout> layout,
                             std::optional<at::Device> device,
                             std::optional<bool> pin_memory) {
  py::gil_scoped_acquire gil;

  py::tuple py_size(size.size());
  for (size_t i = 0; i < size.size(); ++i) {
    py_size[i] = py::int_(size[i]);
  }

  static py::module_ torch_mod = py::module_::import("torch");
  py::object result = torch_mod.attr("zeros")(py_size);
  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(ZerosFn, zeros_stub, FlagosDevice::kFlagOsPython, ZerosKernelPython)

} // namespace at::native::flagos
