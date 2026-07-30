// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized musa stream management for the Moore Threads MUSA backend.
// All MUSA ops share one default stream per device so that consecutive kernels
// keep their implicit ordering without explicit cross-stream synchronization.

#pragma once

#ifdef USE_MUSA

#include <musa_runtime.h>

#include <mutex>
#include <unordered_map>

namespace at::native::flagos::musa {

// One stream per device: a musa stream belongs to the device that was current
// when it was created, so a single global stream would be wrong as soon as an
// op runs on flagos:1.
inline musaStream_t GetDefaultMusaStream() {
  static std::mutex mutex;
  static std::unordered_map<int, musaStream_t> streams;

  int device = 0;
  if (musaGetDevice(&device) != musaSuccess) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex);
  auto it = streams.find(device);
  if (it != streams.end()) {
    return it->second;
  }
  musaStream_t stream = nullptr;
  if (musaStreamCreate(&stream) != musaSuccess) {
    // Fall back to the null (legacy default) stream; ops still execute, just
    // without a dedicated queue.
    stream = nullptr;
  }
  streams.emplace(device, stream);
  return stream;
}

} // namespace at::native::flagos::musa

#endif // USE_MUSA
