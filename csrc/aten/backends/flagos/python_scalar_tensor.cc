// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../scalar_tensor.h"

#include <pybind11/pybind11.h>
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

py::object ScalarToPythonLocal(const at::Scalar& s) {
  if (s.isFloatingPoint()) return py::float_(s.toDouble());
  if (s.isIntegral(false)) return py::int_(s.toLong());
  if (s.isBoolean()) return py::bool_(s.toBool());
  return py::float_(s.toDouble());
}

at::Tensor ScalarTensorKernelPython(const at::Scalar& s,
                                    std::optional<at::ScalarType> dtype,
                                    std::optional<at::Layout> layout,
                                    std::optional<at::Device> device,
                                    std::optional<bool> pin_memory) {
  py::gil_scoped_acquire gil;
  static py::module_ torch_mod = py::module_::import("torch");
  py::object result = torch_mod.attr("scalar_tensor")(ScalarToPythonLocal(s));
  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(ScalarTensorFn, scalar_tensor_stub, FlagosDevice::kFlagOsPython, ScalarTensorKernelPython)

} // namespace at::native::flagos
