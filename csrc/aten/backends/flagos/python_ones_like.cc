// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../ones_like.h"

#include <pybind11/pybind11.h>
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

at::Tensor OnesLikeKernelPython(const at::Tensor& self,
                                std::optional<at::ScalarType> dtype,
                                std::optional<at::Layout> layout,
                                std::optional<at::Device> device,
                                std::optional<bool> pin_memory,
                                std::optional<at::MemoryFormat> memory_format) {
  py::gil_scoped_acquire gil;
  static py::module_ torch_mod = py::module_::import("torch");
  py::object result = torch_mod.attr("ones_like")(TensorToPythonLocal(self));
  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(OnesLikeFn, ones_like_stub, FlagosDevice::kFlagOsPython, OnesLikeKernelPython)

} // namespace at::native::flagos
