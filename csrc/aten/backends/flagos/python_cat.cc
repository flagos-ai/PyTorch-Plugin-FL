// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../cat.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>

namespace py = pybind11;

namespace at::native::flagos {

namespace {

py::object TensorToPythonLocal(const at::Tensor& t) {
  if (!t.defined()) return py::none();
  PyObject* obj = THPVariable_Wrap(t);
  return py::reinterpret_steal<py::object>(obj);
}

at::Tensor PythonToTensorLocal(const py::object& obj) {
  if (obj.is_none()) return at::Tensor();
  PyObject* raw = obj.ptr();
  TORCH_CHECK(THPVariable_Check(raw), "Expected a Tensor from Python op");
  return THPVariable_Unpack(raw);
}

at::Tensor CatKernelPython(const at::ITensorListRef& tensors, int64_t dim) {
  py::gil_scoped_acquire gil;

  // Build Python list of tensors
  py::list py_tensors;
  for (const at::Tensor& t : tensors) {
    py_tensors.append(TensorToPythonLocal(t));
  }

  static py::module_ ops_module = py::module_::import("flag_gems.ops");
  py::object func = ops_module.attr("cat");
  py::object result = func(py_tensors, py::int_(dim));
  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(CatFn, cat_stub, FlagosDevice::kFlagOsPython, CatKernelPython)

} // namespace at::native::flagos
