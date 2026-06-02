// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <include/flagos.h>
#include "runtime/accelerator/ascend/acl_stream.h"

#include <ATen/ATen.h>
#include <dlfcn.h>
#include <stdexcept>
#include <cstdint>
#include <vector>

#include <acl/acl_base_rt.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>

namespace at::native::flagos::ascend {

inline aclDataType ToAclDataType(at::ScalarType type) {
  switch (type) {
    case at::kDouble:  return ACL_DOUBLE;
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
  // aclCreateTensor may store pointers to these arrays, so they must outlive
  // the aclTensor* and the executor that references it.
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> storage_dims_;

  AclTensorWrapper(const at::Tensor& tensor) {
    if (!tensor.defined()) {
      acl_tensor = nullptr;
      return;
    }

    auto sz = tensor.sizes();
    auto st = tensor.strides();
    sizes_.assign(sz.begin(), sz.end());
    strides_.assign(st.begin(), st.end());

    int64_t offset = tensor.storage_offset();
    aclDataType dtype = ToAclDataType(tensor.scalar_type());
    aclFormat format = ACL_FORMAT_ND;

    int64_t storage_size = static_cast<int64_t>(
        tensor.storage().nbytes() / tensor.element_size());
    storage_dims_ = {storage_size};

    void* storage_ptr = const_cast<void*>(tensor.storage().data());

    acl_tensor = aclCreateTensor(
        sizes_.data(),
        static_cast<uint64_t>(sizes_.size()),
        dtype,
        strides_.data(),
        offset,
        format,
        storage_dims_.data(),
        static_cast<uint64_t>(storage_dims_.size()),
        storage_ptr);
  }

  // ACL executor caches references to aclTensor objects passed to
  // GetWorkspaceSize. Destroying them here causes use-after-free.
  ~AclTensorWrapper() = default;

  const aclTensor* get() const { return acl_tensor; }
};

inline aclrtStream GetCurrentAclStream() {
  return GetDefaultAclStream();
}

inline void* GetOpApiLibHandle() {
  static void* handle = []() -> void* {
    void* h = dlopen("libopapi.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("Failed to load libopapi.so: ") + (err ? err : "unknown error"));
    }
    // Call Init() to initialize libopapi.so
    typedef void (*InitFunc)();
    InitFunc initFunc = reinterpret_cast<InitFunc>(dlsym(h, "Init"));
    if (initFunc) {
      initFunc();
    }
    return h;
  }();
  return handle;
}

inline void* GetOpBaseLibHandle() {
  static void* handle = []() -> void* {
    void* h = dlopen("libnnopbase.so", RTLD_LAZY);
    if (!h) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("Failed to load libnnopbase.so: ") + (err ? err : "unknown error"));
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
  // Storage for the scalar value (must outlive aclScalar*)
  union {
    double d;
    float f;
    int64_t i64;
    int32_t i32;
    int16_t i16;
    int8_t i8;
    uint8_t u8;
    bool b;
    at::Half h;
    at::BFloat16 bf16;
  } value_storage;

  AclScalarWrapper(const at::Scalar& scalar, at::ScalarType dtype) {
    // aclCreateScalar stores a pointer to the value, so we must keep it alive
    switch (dtype) {
      case at::kDouble:
        value_storage.d = scalar.toDouble();
        acl_scalar = aclCreateScalar(&value_storage.d, ACL_DOUBLE);
        break;
      case at::kFloat:
        value_storage.f = scalar.toFloat();
        acl_scalar = aclCreateScalar(&value_storage.f, ACL_FLOAT);
        break;
      case at::kHalf:
        value_storage.h = static_cast<at::Half>(scalar.toFloat());
        acl_scalar = aclCreateScalar(&value_storage.h, ACL_FLOAT16);
        break;
      case at::kBFloat16:
        value_storage.bf16 = static_cast<at::BFloat16>(scalar.toFloat());
        acl_scalar = aclCreateScalar(&value_storage.bf16, ACL_BF16);
        break;
      case at::kLong:
        value_storage.i64 = scalar.toLong();
        acl_scalar = aclCreateScalar(&value_storage.i64, ACL_INT64);
        break;
      case at::kInt:
        value_storage.i32 = static_cast<int32_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.i32, ACL_INT32);
        break;
      case at::kShort:
        value_storage.i16 = static_cast<int16_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.i16, ACL_INT16);
        break;
      case at::kChar:
        value_storage.i8 = static_cast<int8_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.i8, ACL_INT8);
        break;
      case at::kByte:
        value_storage.u8 = static_cast<uint8_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.u8, ACL_UINT8);
        break;
      case at::kBool:
        value_storage.b = scalar.toBool();
        acl_scalar = aclCreateScalar(&value_storage.b, ACL_BOOL);
        break;
      default:
        TORCH_CHECK(false, "Unsupported scalar type for ACL: ", dtype);
    }
  }

  ~AclScalarWrapper() = default;

  const aclScalar* get() const { return acl_scalar; }
};

struct AclIntArrayWrapper {
  aclIntArray* acl_array = nullptr;

  AclIntArrayWrapper(at::IntArrayRef arr) {
    acl_array = aclCreateIntArray(arr.data(), arr.size());
  }

  ~AclIntArrayWrapper() = default;

  const aclIntArray* get() const { return acl_array; }
};

struct AclTensorListWrapper {
  aclTensorList* acl_list = nullptr;

  AclTensorListWrapper(const std::vector<const aclTensor*>& tensors) {
    acl_list = aclCreateTensorList(tensors.data(), tensors.size());
  }

  ~AclTensorListWrapper() = default;

  void release() { acl_list = nullptr; }

  const aclTensorList* get() const { return acl_list; }
};

} // namespace at::native::flagos::ascend

#define EXEC_ASCEND_CMD(aclnn_api, ...)                                       \
  do {                                                                        \
    static void* opApiFuncAddr = nullptr;                                     \
    static void* getWorkspaceSizeFuncAddr = nullptr;                          \
    at::native::flagos::ascend::GetApiFunc(                                   \
        #aclnn_api, #aclnn_api "GetWorkspaceSize",                            \
        opApiFuncAddr, getWorkspaceSizeFuncAddr);                              \
                                                                              \
    auto acl_stream = at::native::flagos::ascend::GetCurrentAclStream();      \
                                                                              \
    uint64_t workspace_size = 0;                                              \
    aclOpExecutor* executor = nullptr;                                        \
                                                                              \
    TORCH_CHECK(getWorkspaceSizeFuncAddr && opApiFuncAddr,                     \
        "Failed to load symbols for " #aclnn_api ": ", dlerror());            \
                                                                              \
    typedef int (*GetWorkspaceSizeFunc)(...);                                  \
    auto getWorkspaceSize =                                                   \
        reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr);      \
    int ws_ret = getWorkspaceSize(__VA_ARGS__, &workspace_size, &executor);    \
    TORCH_CHECK(ws_ret == 0,                                                  \
        #aclnn_api "GetWorkspaceSize failed, ret=", ws_ret);                  \
                                                                              \
    void* workspace_addr = nullptr;                                           \
    at::Tensor workspace_tensor;                                              \
    if (workspace_size > 0) {                                                 \
      workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)},   \
          at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));    \
      workspace_addr = workspace_tensor.data_ptr();                           \
    }                                                                         \
                                                                              \
    typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);    \
    auto executeFunc = reinterpret_cast<ExecFunc>(opApiFuncAddr);             \
    int exec_ret = executeFunc(                                               \
        workspace_addr, workspace_size, executor, acl_stream);                \
    TORCH_CHECK(exec_ret == 0, #aclnn_api " execution failed, ret=",         \
        exec_ret);                                                            \
    aclrtSynchronizeStream(acl_stream);                                       \
  } while (false)

