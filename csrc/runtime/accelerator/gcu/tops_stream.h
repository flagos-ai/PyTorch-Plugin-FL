// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized tops stream management for the Enflame GCU backend.
// All GCU ops share one default stream per device so that consecutive kernels
// keep their implicit ordering without explicit cross-stream synchronization.

#pragma once

#ifdef USE_GCU

#include <tops_runtime_api.h>

#include <mutex>
#include <unordered_map>

namespace at::native::flagos::gcu {

// One stream per device: a tops stream belongs to the device that was current
// when it was created, so a single global stream would be wrong as soon as an
// op runs on flagos:1.
inline topsStream_t GetDefaultTopsStream() {
  static std::mutex mutex;
  static std::unordered_map<int, topsStream_t> streams;

  int device = 0;
  if (topsGetDevice(&device) != topsSuccess) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex);
  auto it = streams.find(device);
  if (it != streams.end()) {
    return it->second;
  }
  topsStream_t stream = nullptr;
  if (topsStreamCreate(&stream) != topsSuccess) {
    // Fall back to the null (legacy default) stream; ops still execute, just
    // without a dedicated queue.
    stream = nullptr;
  }
  streams.emplace(device, stream);
  return stream;
}

} // namespace at::native::flagos::gcu

#endif // USE_GCU
