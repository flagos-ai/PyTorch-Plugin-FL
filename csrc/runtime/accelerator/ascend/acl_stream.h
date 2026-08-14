// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized ACL stream management for the Ascend backend.
// All Ascend ops (both native aclnn and FlagGems/TritonJIT) share the same
// default stream to ensure implicit ordering without explicit synchronization.

#pragma once

#ifdef USE_ASCEND

#include <acl/acl_rt.h>

#include <macros.h>

namespace at::native::flagos::ascend {

// Returns the process-wide default ACL stream that ALL Ascend ops share.
//
// This MUST be a single external-linkage, default-visibility symbol defined
// once (in libflagos.so). It used to be an `inline` function with a
// function-local `static`; under -fvisibility=hidden that produced a SEPARATE
// stream instance per shared object (libflagos.so vs libtorch_fl.so). The aten
// kernels in libtorch_fl.so then enqueued ops on one stream while the
// drain-before-read in libflagos.so's memory.cc synchronized a DIFFERENT
// stream, so host-visible D2H reads never waited for the producing kernels ->
// silent corruption under async dispatch. Keep it a plain exported function.
FLAGOS_EXPORT aclrtStream GetDefaultAclStream();

// Return the stream selected for the calling thread and device. If no
// auxiliary stream has been selected, this returns the shared default stream.
FLAGOS_EXPORT aclrtStream GetCurrentAclStream();
FLAGOS_EXPORT aclrtStream GetCurrentAclStreamForDevice(int device_index);
FLAGOS_EXPORT void SetCurrentAclStream(aclrtStream stream);
FLAGOS_EXPORT void SetCurrentAclStreamForDevice(int device_index, aclrtStream stream);

} // namespace at::native::flagos::ascend

#endif // USE_ASCEND
