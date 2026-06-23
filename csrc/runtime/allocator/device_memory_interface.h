// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstddef>

#include <include/flagos.h>

namespace c10::flagos {

// Abstract interface for device memory operations.
// Each hardware backend (CUDA, Metax, Ascend) implements this interface
// to provide raw memory allocation and synchronization primitives.
struct DeviceMemoryInterface {
  virtual ~DeviceMemoryInterface() = default;

  // Raw device memory allocation (e.g., cudaMalloc)
  virtual Error_t device_malloc(void** ptr, size_t size) = 0;
  // Raw device memory free (e.g., cudaFree)
  virtual Error_t device_free(void* ptr) = 0;

  // Device management
  virtual Error_t get_device_index(int* device) = 0;
  virtual Error_t set_device(int device) = 0;

  // Memory info query (free / total bytes on current device)
  virtual Error_t get_memory_info(size_t* free, size_t* total) = 0;

  // Event-based synchronization for stream safety
  virtual Error_t event_create(Event_t* event) = 0;
  virtual Error_t event_destroy(Event_t event) = 0;
  virtual Error_t event_record(Event_t event, Stream_t stream) = 0;
  // Returns Success if event has completed, ErrorNotReady otherwise
  virtual Error_t event_query(Event_t event) = 0;

  // Memcpy for block data migration
  virtual Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) = 0;
};

} // namespace c10::flagos
