// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../device_boxing.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/core/ivalue.h>

#include <mutex>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;

namespace at::native::flagos {

namespace {

struct PythonOpCache {
  py::module_ ops_module;
  std::unordered_map<std::string, py::object> func_cache;
  bool initialized = false;
  std::mutex init_mutex;

  void EnsureInitialized() {
    if (initialized) return;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized) return;
    py::gil_scoped_acquire gil;
    ops_module = py::module_::import("flag_gems.ops");
    initialized = true;
  }

  py::object GetFunc(const char* name) {
    auto it = func_cache.find(name);
    if (it != func_cache.end()) return it->second;
    py::object func = ops_module.attr(name);
    func_cache[name] = func;
    return func;
  }
};

PythonOpCache& GetCache() {
  // Intentionally leaked: prevent destructor from running after Python
  // interpreter finalization, which would crash on Py_DECREF with no GIL.
  static PythonOpCache* cache = new PythonOpCache();
  return *cache;
}

// Convert at::Tensor to Python THPVariable (borrowed ref wrapped in py::object)
// CPU scalar tensors (0-dim) are moved to CUDA since FlagGems kernels
// cannot access CPU memory. PrivateUse1 tensors are assumed to be already
// boxed to CUDA by the caller (via DeviceBoxingGuard).
py::object TensorToPython(const at::Tensor& t) {
  if (!t.defined()) return py::none();
  if (t.device().is_cpu() && t.dim() == 0) {
    auto cuda_t = t.to(c10::Device(c10::DeviceType::CUDA, 0));
    PyObject* obj = THPVariable_Wrap(cuda_t);
    return py::reinterpret_steal<py::object>(obj);
  }
  PyObject* obj = THPVariable_Wrap(t);
  return py::reinterpret_steal<py::object>(obj);
}

// Convert Python THPVariable back to at::Tensor
// Unboxes device back to flagos (PrivateUse1).
at::Tensor PythonToTensor(const py::object& obj) {
  if (obj.is_none()) return at::Tensor();
  PyObject* raw = obj.ptr();
  TORCH_CHECK(THPVariable_Check(raw), "Expected a Tensor from Python op");
  at::Tensor result = THPVariable_Unpack(raw);
  if (result.defined() && result.device().type() == c10::DeviceType::CUDA) {
    UnboxToFlagos(result);
  }
  return result;
}

// Convert at::Scalar to Python
py::object ScalarToPython(const at::Scalar& s) {
  if (s.isFloatingPoint()) {
    return py::float_(s.toDouble());
  } else if (s.isIntegral(/*includeBool=*/false)) {
    return py::int_(s.toLong());
  } else if (s.isBoolean()) {
    return py::bool_(s.toBool());
  } else if (s.isComplex()) {
    auto c = s.toComplexDouble();
    return py::cast(c);
  }
  return py::float_(s.toDouble());
}

// Convert IntArrayRef to Python tuple
py::object IntArrayRefToPython(at::IntArrayRef arr) {
  py::tuple t(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    t[i] = py::int_(arr[i]);
  }
  return t;
}

// Convert OptionalIntArrayRef to Python
py::object OptionalIntArrayRefToPython(at::OptionalIntArrayRef arr) {
  if (!arr.has_value()) return py::none();
  return IntArrayRefToPython(*arr);
}

// Convert optional<ScalarType> to Python
py::object OptionalDtypeToPython(std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) return py::none();
  // Import torch and get the dtype object
  static py::module_ torch_mod = py::module_::import("torch");
  switch (*dtype) {
    case at::ScalarType::Float:   return torch_mod.attr("float32");
    case at::ScalarType::Double:  return torch_mod.attr("float64");
    case at::ScalarType::Half:    return torch_mod.attr("float16");
    case at::ScalarType::BFloat16: return torch_mod.attr("bfloat16");
    case at::ScalarType::Int:     return torch_mod.attr("int32");
    case at::ScalarType::Long:    return torch_mod.attr("int64");
    case at::ScalarType::Short:   return torch_mod.attr("int16");
    case at::ScalarType::Byte:    return torch_mod.attr("uint8");
    case at::ScalarType::Char:    return torch_mod.attr("int8");
    case at::ScalarType::Bool:    return torch_mod.attr("bool");
    default: return py::none();
  }
}

} // namespace

at::Tensor CallPythonOp_T(const char* func_name, const at::Tensor& self) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(self);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self));
  return PythonToTensor(result);
}

at::Tensor& CallPythonOp_T_inplace(const char* func_name, at::Tensor& self) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(self);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  func(TensorToPython(self));
  return self;
}

at::Tensor CallPythonOp_TT(const char* func_name, const at::Tensor& a, const at::Tensor& b) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(a, b);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TTS(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(a, b);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b), "alpha"_a = ScalarToPython(alpha));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TS(const char* func_name, const at::Tensor& self, const at::Scalar& other) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(self);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self), ScalarToPython(other));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TIB(const char* func_name, const at::Tensor& self, int64_t dim, bool flag) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(self);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self), py::int_(dim), py::bool_(flag));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TOIB(const char* func_name, const at::Tensor& self,
                              at::OptionalIntArrayRef dim, bool keepdim,
                              std::optional<at::ScalarType> dtype) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(self);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(
      TensorToPython(self),
      OptionalIntArrayRefToPython(dim),
      py::bool_(keepdim),
      OptionalDtypeToPython(dtype));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TTT(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  DeviceBoxingGuard guard(a, b, c);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b), TensorToPython(c));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_Generic(const char* func_name, const std::vector<c10::IValue>& args) {
  auto& cache = GetCache();
  cache.EnsureInitialized();

  // Collect all tensor args for boxing guard
  std::vector<at::Tensor> tensors;
  for (const auto& val : args) {
    if (val.isTensor() && val.toTensor().defined()) {
      tensors.push_back(val.toTensor());
    }
  }
  for (auto& t : tensors) {
    if (t.device().type() == c10::DeviceType::PrivateUse1) {
      BoxToCuda(t);
    }
  }

  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);

  py::tuple py_args(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& val = args[i];
    if (val.isTensor()) {
      py_args[i] = TensorToPython(val.toTensor());
    } else if (val.isInt()) {
      py_args[i] = py::int_(val.toInt());
    } else if (val.isDouble()) {
      py_args[i] = py::float_(val.toDouble());
    } else if (val.isBool()) {
      py_args[i] = py::bool_(val.toBool());
    } else if (val.isNone()) {
      py_args[i] = py::none();
    } else if (val.isIntList()) {
      auto list = val.toIntList();
      py::tuple t(list.size());
      for (size_t j = 0; j < list.size(); ++j) {
        t[j] = py::int_(static_cast<int64_t>(list[j]));
      }
      py_args[i] = t;
    } else if (val.isScalar()) {
      py_args[i] = ScalarToPython(val.toScalar());
    } else {
      TORCH_CHECK(false, "Unsupported IValue type in CallPythonOp_Generic for op: ", func_name);
    }
  }

  py::object result = func(*py_args);

  // Unbox tensors back to flagos
  for (auto& t : tensors) {
    if (t.device().type() == c10::DeviceType::CUDA) {
      UnboxToFlagos(t);
    }
  }

  return PythonToTensor(result);
}

} // namespace at::native::flagos
