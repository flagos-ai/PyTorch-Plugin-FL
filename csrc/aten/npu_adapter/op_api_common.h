// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <include/flagos.h>

#include <dlfcn.h>
#include <stdexcept>
#include <cstdint>

typedef void aclOpExecutor;
typedef void* aclrtStream;

namespace at::native::flagos::npu {

inline aclrtStream GetCurrentAclStream() {
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

} // namespace at::native::flagos::npu

#define EXEC_NPU_CMD(aclnn_api, ...)                                          \
  do {                                                                        \
    static void* opApiFuncAddr = nullptr;                                     \
    static void* getWorkspaceSizeFuncAddr = nullptr;                          \
    at::native::flagos::npu::GetApiFunc(                                      \
        #aclnn_api, #aclnn_api "GetWorkspaceSize",                            \
        opApiFuncAddr, getWorkspaceSizeFuncAddr);                              \
                                                                              \
    auto acl_stream = at::native::flagos::npu::GetCurrentAclStream();         \
                                                                              \
    uint64_t workspace_size = 0;                                              \
    aclOpExecutor* executor = nullptr;                                        \
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
  } while (false)
