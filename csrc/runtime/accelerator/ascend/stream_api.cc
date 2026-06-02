// Copyright (c) 2026, BAAI. All rights reserved.
//
// Public C API for FlagGems to obtain the current ACL stream managed by torch_fl.
// FlagGems calls this from backend_utils.h::getRawStream() to get the aclrtStream
// for passing to TritonJIT kernels.
//
// Only compiled on Ascend builds (USE_ASCEND=1).

#ifdef USE_ASCEND

#include "acl_stream.h"

extern "C" {

__attribute__((visibility("default")))
void* GetCurrentStream(int device_index) {
  (void)device_index;
  return (void*)at::native::flagos::ascend::GetDefaultAclStream();
}

} // extern "C"

#endif // USE_ASCEND
