// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized ACL stream management for the Ascend backend.
// All Ascend ops (both native aclnn and FlagGems/TritonJIT) share the same
// default stream to ensure implicit ordering without explicit synchronization.

#pragma once

#ifdef USE_ASCEND

#include <acl/acl_rt.h>

namespace at::native::flagos::ascend {

inline aclrtStream GetDefaultAclStream() {
  static aclrtStream stream = []() -> aclrtStream {
    aclrtStream s = nullptr;
    aclrtCreateStream(&s);
    return s;
  }();
  return stream;
}

} // namespace at::native::flagos::ascend

#endif // USE_ASCEND
