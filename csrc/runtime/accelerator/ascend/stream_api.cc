// Copyright (c) 2026, BAAI. All rights reserved.
//
// Public C API for FlagGems to obtain the current ACL stream managed by torch_fl.
// FlagGems calls this from backend_utils.h::getRawStream() to get the aclrtStream
// for passing to TritonJIT kernels.
//
// Only compiled on Ascend builds (USE_ASCEND=1).

#ifdef USE_ASCEND

#include "acl_stream.h"

#include <flagos.h>

#include <unordered_map>

namespace at::native::flagos::ascend {
namespace {

thread_local std::unordered_map<int, aclrtStream> current_streams;

} // namespace

// Single process-wide definition of the shared default ACL stream. Declared in
// acl_stream.h with default visibility so every shared object (libflagos.so and
// libtorch_fl.so) resolves to THIS one instance.
FLAGOS_EXPORT aclrtStream GetDefaultAclStream() {
  static aclrtStream stream = []() -> aclrtStream {
    aclrtStream s = nullptr;
    aclrtCreateStream(&s);
    return s;
  }();
  return stream;
}

FLAGOS_EXPORT aclrtStream GetCurrentAclStreamForDevice(int device_index) {
  auto it = current_streams.find(device_index);
  if (it != current_streams.end() && it->second != nullptr) {
    return it->second;
  }
  return GetDefaultAclStream();
}

FLAGOS_EXPORT aclrtStream GetCurrentAclStream() {
  // ::GetDevice() reads the cached current device (device.cc's gCurrentDevice)
  // instead of round-tripping through aclrtGetDevice. This sits on the dispatch
  // path of every aclnn op, so the runtime call is not affordable here.
  int device_index = 0;
  ::GetDevice(&device_index);
  return GetCurrentAclStreamForDevice(device_index);
}

FLAGOS_EXPORT void SetCurrentAclStreamForDevice(
    int device_index,
    aclrtStream stream) {
  if (stream == nullptr || stream == GetDefaultAclStream()) {
    current_streams.erase(device_index);
  } else {
    current_streams[device_index] = stream;
  }
}

FLAGOS_EXPORT void SetCurrentAclStream(aclrtStream stream) {
  int device_index = 0;
  ::GetDevice(&device_index);
  SetCurrentAclStreamForDevice(device_index, stream);
}

} // namespace at::native::flagos::ascend

extern "C" {

__attribute__((visibility("default")))
void* GetCurrentStream(int device_index) {
  return (void*)at::native::flagos::ascend::GetCurrentAclStreamForDevice(
      device_index);
}

__attribute__((visibility("default")))
void SetCurrentStream(int device_index, void* stream) {
  at::native::flagos::ascend::SetCurrentAclStreamForDevice(
      device_index, (aclrtStream)stream);
}

} // extern "C"

#endif // USE_ASCEND
