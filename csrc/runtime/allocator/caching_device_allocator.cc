// Copyright (c) 2026, BAAI. All rights reserved.

#include "caching_device_allocator.h"

#if !defined(USE_ASCEND)
#include "backends/cuda_memory.h"
#endif

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdlib>

namespace c10::flagos {

// Static singleton pointer for the deleter callback.
CachingDeviceAllocator* CachingDeviceAllocator::instance_ = nullptr;

// Check env var to determine if caching is enabled.
bool CachingDeviceAllocator::is_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("FLAGOS_USE_CACHING_ALLOCATOR");
    if (env && std::string(env) == "0") {
      return false;
    }
    return true;
  }();
  return enabled;
}

CachingDeviceAllocator::CachingDeviceAllocator(
    std::unique_ptr<DeviceMemoryInterface> backend)
    : backend_(std::move(backend)) {
  instance_ = this;
}

CachingDeviceAllocator::~CachingDeviceAllocator() {
  // Release all cached memory on destruction.
  for (auto& state_ptr : device_states_) {
    if (state_ptr) {
      release_cached_blocks(*state_ptr);
    }
  }
  instance_ = nullptr;
}

CachingDeviceAllocator::DeviceState& CachingDeviceAllocator::get_device_state(
    int device) {
  std::lock_guard<std::recursive_mutex> lock(device_states_mutex_);
  if (device >= static_cast<int>(device_states_.size())) {
    device_states_.resize(device + 1);
  }
  if (!device_states_[device]) {
    device_states_[device] = std::make_unique<DeviceState>();
  }
  return *device_states_[device];
}

at::DataPtr CachingDeviceAllocator::allocate(size_t nbytes) {
  if (nbytes == 0) {
    return {nullptr, nullptr, &block_deleter,
            c10::Device(c10::DeviceType::PrivateUse1, 0)};
  }

  int device = -1;
  backend_->get_device_index(&device);
  TORCH_CHECK(device >= 0, "CachingDeviceAllocator: invalid device index");

  // Use nullptr as stream for default stream allocations.
  // In a more complete implementation, we would get the current stream.
  Stream_t stream = nullptr;

  Block* block = alloc_block(nbytes, stream, device);
  TORCH_CHECK(
      block != nullptr,
      "CachingDeviceAllocator: failed to allocate ",
      nbytes,
      " bytes on device ",
      device);

  auto curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device));
  return {block->ptr, block->ptr, &block_deleter, curr_device};
}

at::DeleterFnPtr CachingDeviceAllocator::raw_deleter() const {
  return &block_deleter;
}

void CachingDeviceAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  backend_->memcpy(dest, src, count, MemcpyDeviceToDevice);
}

// --- Core allocation logic ---

Block* CachingDeviceAllocator::alloc_block(
    size_t size,
    Stream_t stream,
    int device) {
  size_t alloc_size = round_size(size);
  auto& state = get_device_state(device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);

  // Process any completed events first to reclaim blocks.
  process_events(state);

  // Select pool based on size.
  BlockPool& pool =
      (alloc_size <= kSmallSize) ? state.small_blocks : state.large_blocks;

  // Try to find a free block in the pool.
  Block* block = find_free_block(alloc_size, stream, pool, device);

  if (!block) {
    // No suitable cached block found, allocate from device.
    // First attempt:
    if (!alloc_from_device(alloc_size, stream, device, &block)) {
      // OOM - try to free cached blocks and retry.
      process_events(state);
      release_cached_blocks(state);
      state.stats.num_alloc_retries++;

      if (!alloc_from_device(alloc_size, stream, device, &block)) {
        // Still OOM after releasing cache.
        return nullptr;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(block != nullptr && block->ptr != nullptr);

  // Split block if significantly larger than needed.
  try_split_block(block, alloc_size);

  block->allocated = true;
  block->requested_size = size;
  state.active_blocks.insert(block);

  // Update stats.
  state.stats.bytes_allocated += block->size;
  state.stats.peak_allocated =
      std::max(state.stats.peak_allocated, state.stats.bytes_allocated);
  state.stats.num_alloc_calls++;

  // Register in ptr-to-block map.
  {
    std::lock_guard<std::mutex> ptr_lock(ptr_map_mutex_);
    ptr_to_block_[block->ptr] = block;
  }

  return block;
}

void CachingDeviceAllocator::free_block(Block* block) {
  auto& state = get_device_state(block->device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);

  block->allocated = false;
  state.active_blocks.erase(block);

  // Update stats.
  state.stats.bytes_allocated -= block->size;
  state.stats.num_free_calls++;

  // Remove from ptr map.
  {
    std::lock_guard<std::mutex> ptr_lock(ptr_map_mutex_);
    ptr_to_block_.erase(block->ptr);
  }

  // If there are outstanding events on other streams, defer the free.
  if (block->event_count > 0) {
    // Block will be returned to pool when events complete.
    return;
  }

  // Try to merge with adjacent free blocks.
  Block* merged = try_merge_blocks(block, state);

  // Return block to its original pool.
  // Use the pool pointer stored on the block (set at allocation time)
  // rather than re-classifying by size. This ensures that a small-pool
  // block that grows after merging remains findable from the small pool.
  TORCH_INTERNAL_ASSERT(merged->pool != nullptr);
  merged->pool->blocks.insert(merged);
}

Block* CachingDeviceAllocator::find_free_block(
    size_t size,
    Stream_t stream,
    BlockPool& pool,
    int device) {
  // Create a search key.
  Block search_key(device, stream, size);
  auto it = pool.blocks.lower_bound(&search_key);

  if (it == pool.blocks.end()) {
    return nullptr;
  }

  Block* block = *it;
  // Must be on the same stream (or we'd need synchronization).
  if (block->stream != stream) {
    return nullptr;
  }

  pool.blocks.erase(it);
  return block;
}

bool CachingDeviceAllocator::alloc_from_device(
    size_t size,
    Stream_t stream,
    int device,
    Block** out) {
  size_t alloc_size = get_allocation_size(size);

  // Ensure device is set correctly.
  backend_->set_device(device);

  void* ptr = nullptr;
  Error_t err = backend_->device_malloc(&ptr, alloc_size);
  if (err != Success || ptr == nullptr) {
    *out = nullptr;
    return false;
  }

  auto& state = get_device_state(device);
  state.stats.num_device_malloc++;
  state.stats.bytes_reserved += alloc_size;
  state.stats.peak_reserved =
      std::max(state.stats.peak_reserved, state.stats.bytes_reserved);

  BlockPool& pool =
      (size <= kSmallSize) ? state.small_blocks : state.large_blocks;
  Block* block = new Block(device, stream, alloc_size, &pool, ptr);
  *out = block;
  return true;
}

void CachingDeviceAllocator::try_split_block(Block* block, size_t size) {
  if (block->size <= size) {
    return;
  }

  size_t remaining = block->size - size;
  if (remaining < kMinBlockSize) {
    return;
  }

  // Create a new block for the remainder.
  Block* remaining_block = new Block(
      block->device,
      block->stream,
      remaining,
      block->pool,
      static_cast<char*>(block->ptr) + size);

  remaining_block->prev = block;
  remaining_block->next = block->next;
  if (block->next) {
    block->next->prev = remaining_block;
  }
  block->next = remaining_block;
  block->size = size;

  // Insert remainder into the pool.
  block->pool->blocks.insert(remaining_block);
}

Block* CachingDeviceAllocator::try_merge_blocks(
    Block* block,
    DeviceState& state) {
  // Merge with next block if it's free.
  if (block->next && !block->next->allocated &&
      block->next->event_count == 0) {
    Block* next = block->next;

    // Remove next from its pool (use pool pointer, not size-based lookup).
    TORCH_INTERNAL_ASSERT(next->pool != nullptr);
    next->pool->blocks.erase(next);

    block->size += next->size;
    block->next = next->next;
    if (next->next) {
      next->next->prev = block;
    }
    delete next;
  }

  // Merge with prev block if it's free.
  if (block->prev && !block->prev->allocated &&
      block->prev->event_count == 0) {
    Block* prev = block->prev;

    // Remove prev from its pool (use pool pointer, not size-based lookup).
    TORCH_INTERNAL_ASSERT(prev->pool != nullptr);
    prev->pool->blocks.erase(prev);

    prev->size += block->size;
    prev->next = block->next;
    if (block->next) {
      block->next->prev = prev;
    }

    // block is now absorbed into prev.
    delete block;
    return prev;
  }

  return block;
}

bool CachingDeviceAllocator::release_cached_blocks(DeviceState& state) {
  bool released = false;

  auto release_pool = [&](BlockPool& pool) {
    std::vector<Block*> to_release;
    std::vector<Block*> to_keep;

    for (auto* block : pool.blocks) {
      if (!block->is_split()) {
        to_release.push_back(block);
      } else {
        to_keep.push_back(block);
      }
    }
    pool.blocks.clear();

    for (auto* block : to_keep) {
      pool.blocks.insert(block);
    }

    for (auto* block : to_release) {
      backend_->device_free(block->ptr);
      state.stats.bytes_reserved -= block->size;
      state.stats.num_device_free++;
      delete block;
      released = true;
    }
  };

  release_pool(state.small_blocks);
  release_pool(state.large_blocks);

  return released;
}

void CachingDeviceAllocator::process_events(DeviceState& state) {
  while (!state.outstanding_events.empty()) {
    auto& [event, block] = state.outstanding_events.front();

    Error_t status = backend_->event_query(event);
    if (status == ErrorNotReady) {
      // Events are ordered; if this one isn't ready, none after it are.
      break;
    }

    // Event completed - destroy it and decrement block's event count.
    backend_->event_destroy(event);
    block->event_count--;

    if (block->event_count == 0 && !block->allocated) {
      // Block is fully released from all streams, return to pool.
      Block* merged = try_merge_blocks(block, state);
      TORCH_INTERNAL_ASSERT(merged->pool != nullptr);
      merged->pool->blocks.insert(merged);
    }

    state.outstanding_events.pop_front();
  }
}

// --- Public API ---

void CachingDeviceAllocator::empty_cache() {
  std::lock_guard<std::recursive_mutex> lock(device_states_mutex_);
  for (auto& state_ptr : device_states_) {
    if (state_ptr) {
      std::lock_guard<std::recursive_mutex> dev_lock(state_ptr->mutex);
      process_events(*state_ptr);
      release_cached_blocks(*state_ptr);
    }
  }
}

void CachingDeviceAllocator::record_stream(
    const at::DataPtr& ptr,
    Stream_t stream) {
  if (!ptr.get()) {
    return;
  }

  Block* block = get_block_from_ptr(ptr.get());
  if (!block) {
    return;
  }

  auto& state = get_device_state(block->device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);

  // If the block is already on this stream, nothing to do.
  if (block->stream == stream) {
    return;
  }

  // Record an event on the given stream. When the event completes,
  // we know the stream is done using this block.
  Event_t event = nullptr;
  Error_t err = backend_->event_create(&event);
  TORCH_CHECK(err == Success, "Failed to create event for record_stream");

  err = backend_->event_record(event, stream);
  TORCH_CHECK(err == Success, "Failed to record event for record_stream");

  block->event_count++;
  state.outstanding_events.push_back({event, block});
}

AllocatorStats CachingDeviceAllocator::get_stats(int device) {
  auto& state = get_device_state(device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  return state.stats;
}

void CachingDeviceAllocator::reset_stats(int device) {
  auto& state = get_device_state(device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  state.stats = AllocatorStats{};
}

Block* CachingDeviceAllocator::get_block_from_ptr(void* ptr) {
  std::lock_guard<std::mutex> lock(ptr_map_mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it != ptr_to_block_.end()) {
    return it->second;
  }
  return nullptr;
}

// Static deleter invoked by DataPtr when a tensor is freed.
void CachingDeviceAllocator::block_deleter(void* ptr) {
  if (!ptr || !instance_) {
    return;
  }
  Block* block = instance_->get_block_from_ptr(ptr);
  if (block) {
    instance_->free_block(block);
  }
}

// --- Global accessor ---

CachingDeviceAllocator* GetCachingAllocator() {
  static std::unique_ptr<CachingDeviceAllocator> alloc;
  static std::once_flag flag;
  std::call_once(flag, []() {
    // Select backend based on build configuration.
    // For now, we always create the CUDA backend when this code is compiled.
    // Metax and Ascend backends will be added later.
#if defined(USE_ASCEND)
    // TODO: create AscendDeviceMemory
    TORCH_CHECK(false, "Caching allocator not yet implemented for Ascend");
#else
    // CUDA (and Metax, which uses CUDA-compatible API)
    auto backend = std::make_unique<CudaDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#endif
  });
  return alloc.get();
}

} // namespace c10::flagos
