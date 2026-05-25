// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../new_ones.h"

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

at::Tensor NewOnesKernelPython(const at::Tensor& self, at::IntArrayRef size,
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
  // new_ones is a Tensor method; call via torch.ones with same dtype/device
  py::object result = torch_mod.attr("ones")(
      py_size,
      py::arg("dtype") = dtype.has_value() ? torch_mod.attr("float32") : py::none(),
      py::arg("device") = TensorToPythonLocal(self).attr("device"));
  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(NewOnesFn, new_ones_stub, FlagosDevice::kFlagOsPython, NewOnesKernelPython)

} // namespace at::native::flagos
