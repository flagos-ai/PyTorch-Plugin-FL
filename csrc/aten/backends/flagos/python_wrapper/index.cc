// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../index.h"

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

at::Tensor IndexTensorKernelPython(const at::Tensor& self, const c10::List<::std::optional<at::Tensor>>& indices) {
  py::gil_scoped_acquire gil;

  // Build Python list of index tensors (None for missing dims)
  py::list py_indices;
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto& idx = indices[i];
    if (idx.has_value()) {
      py_indices.append(TensorToPythonLocal(*idx));
    } else {
      py_indices.append(py::none());
    }
  }

  static py::module_ ops_module = py::module_::import("flag_gems.ops");
  py::object func = ops_module.attr("index");
  py::object result = func(TensorToPythonLocal(self), py::tuple(py_indices));
  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(IndexTensorFn, index_tensor_stub, FlagosDevice::kFlagOsPython, IndexTensorKernelPython)

} // namespace at::native::flagos
