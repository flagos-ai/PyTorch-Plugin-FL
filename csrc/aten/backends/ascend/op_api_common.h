// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <include/flagos.h>

#include <ATen/ATen.h>
#include <dlfcn.h>
#include <stdexcept>
#include <cstdint>
#include <vector>

#include <acl/acl_base_rt.h>
#include <aclnn/acl_meta.h>

namespace at::native::flagos::ascend {

inline aclDataType ToAclDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:   return ACL_FLOAT;
    case at::kHalf:    return ACL_FLOAT16;
    case at::kBFloat16: return ACL_BF16;
    case at::kInt:     return ACL_INT32;
    case at::kLong:    return ACL_INT64;
    case at::kShort:   return ACL_INT16;
    case at::kChar:    return ACL_INT8;
    case at::kByte:    return ACL_UINT8;
    case at::kBool:    return ACL_BOOL;
    default:
      TORCH_CHECK(false, "Unsupported dtype for ACL: ", type);
  }
}

struct AclTensorWrapper {
  aclTensor* acl_tensor = nullptr;

  AclTensorWrapper(const at::Tensor& tensor) {
    if (!tensor.defined()) {
      acl_tensor = nullptr;
      return;
    }

    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t offset = tensor.storage_offset();
    aclDataType dtype = ToAclDataType(tensor.scalar_type());
    aclFormat format = ACL_FORMAT_ND;

    int64_t storage_size = static_cast<int64_t>(
        tensor.storage().nbytes() / tensor.element_size());
    std::vector<int64_t> storage_dims = {storage_size};

    void* storage_ptr = const_cast<void*>(tensor.storage().data());

    acl_tensor = aclCreateTensor(
        sizes.data(),
        static_cast<uint64_t>(sizes.size()),
        dtype,
        strides.data(),
        offset,
        format,
        storage_dims.data(),
        static_cast<uint64_t>(storage_dims.size()),
        storage_ptr);
  }

  ~AclTensorWrapper() {
    if (acl_tensor) {
      aclDestroyTensor(acl_tensor);
    }
  }

  // Release ownership without destroying the tensor
  void release() { acl_tensor = nullptr; }

  const aclTensor* get() const { return acl_tensor; }
};

inline aclrtStream GetCurrentAclStream() {
  // After aclrtSetDevice, ACL provides an implicit default stream.
  // nullptr tells ACL to use the default stream on the current device.
  return nullptr;
}

inline void* GetOpApiLibHandle() {
  static void* handle = []() -> void* {
    void* h = dlopen("libopapi.so", RTLD_LAZY);
    if (!h) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("Failed to load libopapi.so: ") + (err ? err : "unknown error"));
    }
    return h;
  }();
  return handle;
}

inline void GetApiFunc(const char* api_name, const char* workspace_name,
                       void*& api_func, void*& workspace_func) {
  void* handle = GetOpApiLibHandle();
  if (!api_func) {
    api_func = dlsym(handle, api_name);
  }
  if (!workspace_func) {
    workspace_func = dlsym(handle, workspace_name);
  }
}


struct AclScalarWrapper {
  aclScalar* acl_scalar = nullptr;

  AclScalarWrapper(const at::Scalar& scalar, at::ScalarType dtype) {
    // aclCreateScalar reads exactly the number of bytes implied by dataType,
    // so the pointed-to value must match that width exactly.
    switch (dtype) {
      case at::kDouble: {
        double val = scalar.toDouble();
        acl_scalar = aclCreateScalar(&val, ACL_DOUBLE);
        break;
      }
      case at::kFloat: {
        float val = scalar.toFloat();
        acl_scalar = aclCreateScalar(&val, ACL_FLOAT);
        break;
      }
      case at::kHalf: {
        at::Half val = static_cast<at::Half>(scalar.toFloat());
        acl_scalar = aclCreateScalar(&val, ACL_FLOAT16);
        break;
      }
      case at::kBFloat16: {
        at::BFloat16 val = static_cast<at::BFloat16>(scalar.toFloat());
        acl_scalar = aclCreateScalar(&val, ACL_BF16);
        break;
      }
      case at::kLong: {
        int64_t val = scalar.toLong();
        acl_scalar = aclCreateScalar(&val, ACL_INT64);
        break;
      }
      case at::kInt: {
        int32_t val = static_cast<int32_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&val, ACL_INT32);
        break;
      }
      case at::kShort: {
        int16_t val = static_cast<int16_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&val, ACL_INT16);
        break;
      }
      case at::kChar: {
        int8_t val = static_cast<int8_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&val, ACL_INT8);
        break;
      }
      case at::kByte: {
        uint8_t val = static_cast<uint8_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&val, ACL_UINT8);
        break;
      }
      case at::kBool: {
        bool val = scalar.toBool();
        acl_scalar = aclCreateScalar(&val, ACL_BOOL);
        break;
      }
      default:
        TORCH_CHECK(false, "Unsupported scalar type for ACL: ", dtype);
    }
  }

  ~AclScalarWrapper() {
    if (acl_scalar) {
      aclDestroyScalar(acl_scalar);
    }
  }

  const aclScalar* get() const { return acl_scalar; }
};

struct AclIntArrayWrapper {
  aclIntArray* acl_array = nullptr;

  AclIntArrayWrapper(at::IntArrayRef arr) {
    acl_array = aclCreateIntArray(arr.data(), arr.size());
  }

  ~AclIntArrayWrapper() {
    if (acl_array) {
      aclDestroyIntArray(acl_array);
    }
  }

  const aclIntArray* get() const { return acl_array; }
};

struct AclTensorListWrapper {
  aclTensorList* acl_list = nullptr;

  AclTensorListWrapper(const std::vector<const aclTensor*>& tensors) {
    acl_list = aclCreateTensorList(tensors.data(), tensors.size());
  }

  ~AclTensorListWrapper() {
    if (acl_list) {
      aclDestroyTensorList(acl_list);
    }
  }

  const aclTensorList* get() const { return acl_list; }
};

} // namespace at::native::flagos::ascend

#define EXEC_ASCEND_CMD(aclnn_api, ...)                                       \
  do {                                                                        \
    static void* opApiFuncAddr = nullptr;                                     \
    static void* getWorkspaceSizeFuncAddr = nullptr;                          \
    static void* initMemAddr = nullptr;                                       \
    static void* unInitMemAddr = nullptr;                                     \
    at::native::flagos::ascend::GetApiFunc(                                   \
        #aclnn_api, #aclnn_api "GetWorkspaceSize",                            \
        opApiFuncAddr, getWorkspaceSizeFuncAddr);                              \
    {                                                                         \
      void* handle = at::native::flagos::ascend::GetOpApiLibHandle();         \
      if (!initMemAddr) {                                                     \
        initMemAddr = dlsym(handle, "InitHugeMemThreadLocal");                \
      }                                                                       \
      if (!unInitMemAddr) {                                                   \
        unInitMemAddr = dlsym(handle, "UnInitHugeMemThreadLocal");            \
      }                                                                       \
    }                                                                         \
                                                                              \
    auto acl_stream = at::native::flagos::ascend::GetCurrentAclStream();      \
                                                                              \
    uint64_t workspace_size = 0;                                              \
    aclOpExecutor* executor = nullptr;                                        \
                                                                              \
    TORCH_CHECK(getWorkspaceSizeFuncAddr && opApiFuncAddr,                     \
        "Failed to load symbols for " #aclnn_api ": ", dlerror());            \
                                                                              \
    typedef void (*InitMemFunc)(void*, bool);                                 \
    typedef void (*UnInitMemFunc)(void*, bool);                               \
    if (initMemAddr) {                                                        \
      reinterpret_cast<InitMemFunc>(initMemAddr)(nullptr, false);             \
    }                                                                         \
                                                                              \
    typedef int (*GetWorkspaceSizeFunc)(...);                                  \
    auto getWorkspaceSize =                                                   \
        reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr);      \
    int ws_ret = getWorkspaceSize(__VA_ARGS__, &workspace_size, &executor);    \
    TORCH_CHECK(ws_ret == 0,                                                  \
        #aclnn_api "GetWorkspaceSize failed, ret=", ws_ret);                  \
                                                                              \
    void* workspace_addr = nullptr;                                           \
    if (workspace_size > 0) {                                                 \
      auto malloc_ret = ::Malloc(&workspace_addr, workspace_size);            \
      TORCH_CHECK(malloc_ret == Success,                                      \
          "Workspace allocation failed for " #aclnn_api);                     \
    }                                                                         \
                                                                              \
    typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);    \
    auto executeFunc = reinterpret_cast<ExecFunc>(opApiFuncAddr);             \
    int exec_ret = executeFunc(                                               \
        workspace_addr, workspace_size, executor, acl_stream);                \
    TORCH_CHECK(exec_ret == 0, #aclnn_api " execution failed, ret=",         \
        exec_ret);                                                            \
                                                                              \
    if (workspace_addr) {                                                     \
      ::Free(workspace_addr);                                                 \
    }                                                                         \
                                                                              \
    if (unInitMemAddr) {                                                      \
      reinterpret_cast<UnInitMemFunc>(unInitMemAddr)(nullptr, false);         \
    }                                                                         \
  } while (false)

