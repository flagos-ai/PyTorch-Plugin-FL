// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../nll_loss.h"

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

std::tuple<at::Tensor, at::Tensor> NllLossForwardKernelPython(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
  py::gil_scoped_acquire gil;

  static py::module_ ops_module = py::module_::import("flag_gems.ops");
  py::object func = ops_module.attr("nll_loss_forward");

  py::object py_weight = weight.has_value() ? TensorToPythonLocal(*weight) : py::none();
  py::object result = func(
      TensorToPythonLocal(self),
      TensorToPythonLocal(target),
      py_weight,
      py::int_(reduction),
      py::int_(ignore_index));

  // Result is a tuple (output, total_weight)
  py::tuple result_tuple = result.cast<py::tuple>();
  return std::make_tuple(
      PythonToTensorLocal(result_tuple[0]),
      PythonToTensorLocal(result_tuple[1]));
}

at::Tensor NllLossBackwardKernelPython(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) {
  py::gil_scoped_acquire gil;

  static py::module_ ops_module = py::module_::import("flag_gems.ops");
  py::object func = ops_module.attr("nll_loss_backward");

  py::object py_weight = weight.has_value() ? TensorToPythonLocal(*weight) : py::none();
  py::object result = func(
      TensorToPythonLocal(grad_output),
      TensorToPythonLocal(self),
      TensorToPythonLocal(target),
      py_weight,
      py::int_(reduction),
      py::int_(ignore_index),
      TensorToPythonLocal(total_weight));

  return PythonToTensorLocal(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(NllLossForwardFn, nll_loss_forward_stub, FlagosDevice::kFlagOsPython, NllLossForwardKernelPython)
FLAGOS_REGISTER_DISPATCH(NllLossBackwardFn, nll_loss_backward_stub, FlagosDevice::kFlagOsPython, NllLossBackwardKernelPython)

} // namespace at::native::flagos
