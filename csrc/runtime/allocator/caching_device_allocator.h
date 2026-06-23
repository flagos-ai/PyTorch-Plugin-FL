// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <include/macros.h>

#include "block.h"
#include "device_memory_interface.h"

#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace c10::flagos {

// Statistics for memory usage on a single device.
struct AllocatorStats {
  size_t bytes_allocated = 0;    // currently allocated by user
  size_t bytes_reserved = 0;     // total held by allocator (allocated + cached)
  size_t peak_allocated = 0;
  size_t peak_reserved = 0;
  size_t num_alloc_calls = 0;
  size_t num_free_calls = 0;
  size_t num_device_malloc = 0;  // actual calls to device_malloc
  size_t num_device_free = 0;    // actual calls to device_free
  size_t num_alloc_retries = 0;  // OOM retries
};

// The caching device allocator for PrivateUse1 devices.
// Maintains per-device free block pools and reuses memory to avoid
// repeated expensive device malloc/free calls.
class FLAGOS_EXPORT CachingDeviceAllocator final : public at::Allocator {
 public:
  explicit CachingDeviceAllocator(
      std::unique_ptr<DeviceMemoryInterface> backend);
  ~CachingDeviceAllocator() override;

  // at::Allocator interface
  at::DataPtr allocate(size_t nbytes) override;
  at::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  // --- Caching-specific public API ---

  // Release all unoccupied cached memory back to the device.
  void empty_cache();

  // Record that a block's memory is used on the given stream.
  // This ensures the block won't be reused until the stream completes.
  void record_stream(const at::DataPtr& ptr, Stream_t stream);

  // Get statistics for a device.
  AllocatorStats get_stats(int device);

  // Reset accumulated statistics for a device.
  void reset_stats(int device);

  // Get the underlying block for a data pointer (nullptr if not found).
  Block* get_block_from_ptr(void* ptr);

  // Whether caching is enabled (controlled by env var).
  static bool is_enabled();

 private:
  // Per-device allocator state, each with its own lock.
  struct DeviceState {
    std::recursive_mutex mutex;
    BlockPool large_blocks{false};
    BlockPool small_blocks{true};
    std::unordered_set<Block*> active_blocks;
    // Events waiting to be processed: (event, block)
    std::deque<std::pair<Event_t, Block*>> outstanding_events;
    AllocatorStats stats;

    DeviceState() = default;
  };

  // Allocate a block from the cache or device.
  Block* alloc_block(size_t size, Stream_t stream, int device);

  // Free a block back to the cache.
  void free_block(Block* block);

  // Try to find a suitable cached block.
  Block* find_free_block(
      size_t size,
      Stream_t stream,
      BlockPool& pool,
      int device);

  // Allocate a new segment from the device and create a block.
  bool alloc_from_device(size_t size, Stream_t stream, int device, Block** out);

  // Try to split a block if it's significantly larger than needed.
  void try_split_block(Block* block, size_t size);

  // Merge a freed block with adjacent free blocks. Returns the surviving block.
  Block* try_merge_blocks(Block* block, DeviceState& state);

  // Release all cached blocks for a device (used during OOM retry).
  bool release_cached_blocks(DeviceState& state);

  // Process completed events and return blocks to the pool.
  void process_events(DeviceState& state);

  // Static deleter function for DataPtr.
  static void block_deleter(void* ptr);

  // Get or create per-device state.
  DeviceState& get_device_state(int device);

  // Map from raw pointer to Block for O(1) lookup.
  std::mutex ptr_map_mutex_;
  std::unordered_map<void*, Block*> ptr_to_block_;

  std::unique_ptr<DeviceMemoryInterface> backend_;
  std::vector<std::unique_ptr<DeviceState>> device_states_;
  std::recursive_mutex device_states_mutex_;

  // Global pointer to the singleton, used by the static deleter.
  static CachingDeviceAllocator* instance_;
};

// Get the global caching allocator instance (creates on first call).
FLAGOS_EXPORT CachingDeviceAllocator* GetCachingAllocator();

} // namespace c10::flagos
