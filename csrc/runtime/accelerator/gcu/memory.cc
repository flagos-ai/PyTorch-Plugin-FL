// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <flagos.h>
#include <tops_runtime_api.h>

#include <cstdio>
#include <map>
#include <mutex>

namespace {

struct Block {
  MemoryType type = MemoryType::MemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
};

// Unlike CUDA's unified addressing, the tops runtime resolves a device pointer
// only against the *current* device: a memcpy touching memory owned by another
// device fails with "not allocated in current device" (and, because callers
// often ignore the status, silently yields zeros). Switch to the owning device
// for the duration of the copy and restore afterwards.
class PointerDeviceGuard {
 public:
  // `dev_ptr` must be the device-side pointer of the transfer, or nullptr for
  // host-only copies.
  explicit PointerDeviceGuard(const void* dev_ptr) {
    if (!dev_ptr)
      return;
    topsPointerAttribute_t attr{};
    if (topsPointerGetAttributes(&attr, dev_ptr) != topsSuccess)
      return;
    if (attr.device < 0)
      return;
    int current = -1;
    if (topsGetDevice(&current) != topsSuccess || current == attr.device)
      return;
    if (topsSetDevice(attr.device) != topsSuccess)
      return;
    prev_device_ = current;
  }

  ~PointerDeviceGuard() {
    if (prev_device_ >= 0)
      topsSetDevice(prev_device_);
  }

  PointerDeviceGuard(const PointerDeviceGuard&) = delete;
  PointerDeviceGuard& operator=(const PointerDeviceGuard&) = delete;

 private:
  int prev_device_ = -1;
};

// Returns the device-side pointer involved in a transfer, or nullptr when the
// copy stays on the host. For device-to-device the source device is used; the
// tops runtime handles the peer side of such a copy itself.
const void* DevicePointerFor(
    MemcpyKind kind,
    const void* dst,
    const void* src) {
  switch (kind) {
    case MemcpyHostToDevice:
      return dst;
    case MemcpyDeviceToHost:
    case MemcpyDeviceToDevice:
      return src;
    default:
      return nullptr;
  }
}

// tops enum values line up with MemcpyKind (0=H2H, 1=H2D, 2=D2H, 3=D2D), but
// map explicitly so a future enum change surfaces here instead of silently
// copying in the wrong direction.
bool toTopsMemcpyKind(MemcpyKind kind, topsMemcpyKind* out) {
  switch (kind) {
    case MemcpyHostToHost:
      *out = topsMemcpyHostToHost;
      return true;
    case MemcpyHostToDevice:
      *out = topsMemcpyHostToDevice;
      return true;
    case MemcpyDeviceToHost:
      *out = topsMemcpyDeviceToHost;
      return true;
    case MemcpyDeviceToDevice:
      *out = topsMemcpyDeviceToDevice;
      return true;
    default:
      return false;
  }
}

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  Error_t allocate(void** ptr, size_t size, MemoryType type) {
    if (!ptr || size == 0)
      return ErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    void* mem = nullptr;
    int current_device = -1;

    if (type == MemoryType::MemoryTypeDevice) {
      GetDevice(&current_device);

      // Ensure the tops device is set correctly before allocation.
      topsError_t set_err = topsSetDevice(current_device);
      if (set_err != topsSuccess) {
        fprintf(stderr, "[flagos-gcu] topsSetDevice(%d) failed: %s\n",
                current_device, topsGetErrorString(set_err));
        return ErrorMemoryAllocation;
      }

      topsError_t err = topsMalloc(&mem, size);
      if (err != topsSuccess || mem == nullptr) {
        fprintf(stderr, "[flagos-gcu] topsMalloc(%zu bytes) on device %d failed: %s\n",
                size, current_device, topsGetErrorString(err));
        return ErrorMemoryAllocation;
      }
    } else {
      topsError_t err = topsHostMalloc(&mem, size, topsHostMallocDefault);
      if (err != topsSuccess || mem == nullptr)
        return ErrorMemoryAllocation;
    }

    m_registry[mem] = {type, current_device, mem, size};
    *ptr = mem;
    return Success;
  }

  Error_t free(void* ptr) {
    if (!ptr)
      return Success;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end())
      return ErrorUnknown;

    const auto& info = it->second;
    topsError_t err;
    if (info.type == MemoryType::MemoryTypeDevice) {
      PointerDeviceGuard guard(info.pointer);
      err = topsFree(info.pointer);
    } else {
      err = topsHostFree(info.pointer);
    }

    m_registry.erase(it);
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
    if (!dst || !src || count == 0)
      return ErrorUnknown;

    topsMemcpyKind tops_kind;
    if (!toTopsMemcpyKind(kind, &tops_kind))
      return ErrorUnknown;

    PointerDeviceGuard guard(DevicePointerFor(kind, dst, src));
    topsError_t err = topsMemcpy(dst, src, count, tops_kind);
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind,
      Stream_t stream) {
    if (!dst || !src || count == 0)
      return ErrorUnknown;

    topsMemcpyKind tops_kind;
    if (!toTopsMemcpyKind(kind, &tops_kind))
      return ErrorUnknown;

    PointerDeviceGuard guard(DevicePointerFor(kind, dst, src));
    topsError_t err = topsMemcpyAsync(
        dst, src, count, tops_kind, reinterpret_cast<topsStream_t>(stream));
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t getPointerAttributes(
      PointerAttributes* attributes,
      const void* ptr) {
    if (!attributes || !ptr)
      return ErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);

    if (!info) {
      attributes->type = MemoryType::MemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->pointer;
    }

    return Success;
  }

  Error_t memset(void* devPtr, int value, size_t count) {
    PointerDeviceGuard guard(devPtr);
    topsError_t err = topsMemset(devPtr, value, count);
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t memsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
    PointerDeviceGuard guard(devPtr);
    topsError_t err = topsMemsetAsync(
        devPtr, value, count, reinterpret_cast<topsStream_t>(stream));
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

 private:
  MemoryManager() = default;

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p_char = static_cast<const char*>(ptr);
      const char* base_char = static_cast<const char*>(it->first);
      if (p_char >= base_char && p_char < (base_char + it->second.size)) {
        return &it->second;
      }
    }

    return nullptr;
  }

  std::map<void*, Block> m_registry;
  std::mutex m_mutex;
};

} // namespace

Error_t Malloc(void** devPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      devPtr, size, MemoryType::MemoryTypeDevice);
}

Error_t Free(void* devPtr) {
  return MemoryManager::getInstance().free(devPtr);
}

Error_t MallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, MemoryType::MemoryTypeHost);
}

Error_t FreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
}

Error_t Memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

Error_t MemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    MemcpyKind kind,
    Stream_t stream) {
  return MemoryManager::getInstance().memcpyAsync(dst, src, count, kind, stream);
}

Error_t PointerGetAttributes(PointerAttributes* attributes, const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

Error_t Memset(void* devPtr, int value, size_t count) {
  return MemoryManager::getInstance().memset(devPtr, value, count);
}

Error_t MemsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
  return MemoryManager::getInstance().memsetAsync(devPtr, value, count, stream);
}
