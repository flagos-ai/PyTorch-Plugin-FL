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

#include "flagos_cupti_profiler.h"
#include "cupti_shim.h"

#include <kineto/ActivityType.h>
#include <kineto/ILoggerObserver.h>
#include <kineto/libkineto.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

// Diagnostic logging is gated behind FLAGOS_CUPTI_SHIM_DEBUG=1 so that normal
// profiling runs stay quiet (these callbacks fire once per buffer/session).
namespace {
inline bool flagos_cupti_debug() {
  static const bool on = (std::getenv("FLAGOS_CUPTI_SHIM_DEBUG") != nullptr);
  return on;
}
}  // namespace
#define FLAGOS_CUPTI_LOG(expr) \
  do { if (flagos_cupti_debug()) { std::cerr << expr; } } while (0)

// CUPTI activity record layouts. We mirror the cu12 runtime's
// cupti_activity.h EXACTLY (verified against nvidia-cuda-cupti-cu12's
// CUpti_ActivityKernel9 / CUpti_ActivityMemcpy6): the records are
// __attribute__((packed)) with no natural-alignment padding, `kind` is a
// 4-byte enum, and the kernel `name` is a `const char*` pointer field located
// deep in the struct -- NOT an inline string at a small offset. An earlier
// hand-guessed layout (uint8_t kind + pad[7], name at +56) decoded garbage
// (empty names, zero durations), so these must stay byte-accurate.
//
// CUpti_ActivityKind is a 4-byte enum on this platform.
using CUpti_ActivityKind_t = uint32_t;

#pragma pack(push, 1)
struct CUpti_Activity {
  CUpti_ActivityKind_t kind;
};

// Prefix of CUpti_ActivityKernel9 up to and including `name`. Field order and
// widths are copied verbatim from the cu12 header; the packed attribute makes
// offsets match the runtime records byte-for-byte.
struct CUpti_ActivityKernel9_Compat {
  CUpti_ActivityKind_t kind;
  uint8_t cacheConfig;            // union { uint8_t both; ... } cacheConfig
  uint8_t sharedMemoryConfig;
  uint16_t registersPerThread;
  uint32_t partitionedGlobalCacheRequested;  // enum, 4 bytes
  uint32_t partitionedGlobalCacheExecuted;   // enum, 4 bytes
  uint64_t start;
  uint64_t end;
  uint64_t completed;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
  int32_t blockX;
  int32_t blockY;
  int32_t blockZ;
  int32_t staticSharedMemory;
  int32_t dynamicSharedMemory;
  uint32_t localMemoryPerThread;
  uint32_t localMemoryTotal;
  uint32_t correlationId;
  int64_t gridId;
  const char* name;
  // ... remaining fields omitted (we only read up to `name`).
};

// Prefix of CUpti_ActivityMemcpy6 up to correlationId.
struct CUpti_ActivityMemcpy_Compat {
  CUpti_ActivityKind_t kind;
  uint8_t copyKind;
  uint8_t srcKind;
  uint8_t dstKind;
  uint8_t flags;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
};
#pragma pack(pop)

namespace c10 {
namespace flagos {

namespace detail {

// --- record-layout self-check ------------------------------------------------
//
// The structs above mirror ONE CUPTI version's layout by hand. NVIDIA adds
// fields and publishes a new CUpti_ActivityKernel<N> every few releases, so on a
// different CUPTI the mirror can be wrong -- and the failure mode is silent: we
// decode whatever bytes land at those offsets and emit a trace full of empty
// names and zero durations (exactly the bug this file was fixed for once).
//
// So rather than trust the mirror, check the values it produces. A correct
// decode has properties that garbage almost never satisfies, and each check is
// version-independent -- it asserts something true of any sane kernel record,
// not something true of cu12 specifically:
//
//   * end >= start                  (a kernel cannot finish before it starts)
//   * duration is not absurd        (CUPTI timestamps are ns since boot; a
//                                    multi-hour single kernel means the offsets
//                                    are misaligned, not a slow kernel)
//   * start is a plausible boot-relative ns timestamp, i.e. non-zero
//   * name, if non-null, points at readable non-empty text
//
// On failure we drop the record and report once, naming the bound CUPTI version,
// so the user gets "your CUPTI layout is unsupported" instead of a mysteriously
// empty timeline.

// A single kernel longer than this means we are reading the wrong offsets. Real
// kernels run micro- to milliseconds; an hour is ~7 orders of magnitude beyond
// anything legitimate, so this rejects garbage without risking a false positive
// on a genuinely slow kernel.
constexpr uint64_t kMaxPlausibleDurationNs = 3600ull * 1000 * 1000 * 1000;

std::atomic<uint64_t> g_layout_reject_count{0};
std::once_flag g_layout_warn_once;

// Reports the first rejected record, then stays quiet: a bad layout rejects
// every record, and one diagnostic is informative where thousands are noise.
void reportLayoutMismatch(const char* what) {
  auto& shim = CuptiShim::get();
  std::call_once(g_layout_warn_once, [&] {
    std::cerr
        << "[flagos] CUPTI activity records failed the layout self-check ("
        << what << ").\n"
        << "[flagos]   bound CUPTI: "
        << (shim.library_path[0] ? shim.library_path : "<unknown>")
        << " (API version " << shim.api_version << ")\n"
        << "[flagos]   This build mirrors the CUpti_ActivityKernel9 /"
           " CUpti_ActivityMemcpy6 layouts by hand, so a CUPTI whose record\n"
        << "[flagos]   layout differs cannot be decoded. GPU kernel events will"
           " be missing from the trace; CPU-side profiling is unaffected.\n"
        << "[flagos]   Set FLAGOS_CUPTI_LIBRARY to a matching libcupti, or"
           " report this CUPTI version so its layout can be added.\n";
  });
  g_layout_reject_count.fetch_add(1, std::memory_order_relaxed);
}

// True when a decoded (start, end) pair is self-consistent.
bool timestampsPlausible(uint64_t start, uint64_t end) {
  if (start == 0 || end < start) {
    return false;
  }
  return (end - start) <= kMaxPlausibleDurationNs;
}

// True when `name` is either absent or points at readable non-empty text.
// A misaligned layout usually yields a wild pointer here; probing it with a
// short bounded read is far cheaper than the SIGSEGV that blind trust invites.
bool kernelNamePlausible(const char* name) {
  if (name == nullptr) {
    return true;  // absent is legal; we substitute a placeholder below
  }
  constexpr size_t kMaxNameLen = 4096;
  size_t len = strnlen(name, kMaxNameLen);
  return len > 0 && len < kMaxNameLen;
}


// Global session pointer for buffer callbacks (CUPTI callbacks are C-style,
// cannot capture context).
FlagosCuptiProfilerSession* g_active_session = nullptr;
std::mutex g_session_mutex;

// Instrumentation counters for correlation push/pop verification
std::atomic<uint64_t> g_correlation_push_count{0};
std::atomic<uint64_t> g_correlation_pop_count{0};

// Buffer pool for CUPTI activity records
constexpr size_t kBufferSize = 8 * 1024 * 1024;  // 8MB per buffer
constexpr size_t kBufferAlignment = 8;

void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  FLAGOS_CUPTI_LOG("[flagos] CUPTI bufferRequested callback invoked\n");
  *buffer = (uint8_t*)aligned_alloc(kBufferAlignment, kBufferSize);
  *size = kBufferSize;
  *maxNumRecords = 0;  // no limit
}

void bufferCompleted(
    CUcontext context,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  FLAGOS_CUPTI_LOG("[flagos] CUPTI bufferCompleted callback invoked, validSize=" << validSize << "\n");
  std::lock_guard<std::mutex> lock(g_session_mutex);
  if (!g_active_session || !buffer) {
    if (buffer) {
      free(buffer);
    }
    return;
  }

  auto& shim = CuptiShim::get();
  if (!shim.ok) {
    free(buffer);
    return;
  }

  CUpti_Activity* record = nullptr;
  while (true) {
    CUptiResult status = shim.ActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;  // No more records
    }
    if (status != CUPTI_SUCCESS || !record) {
      break;
    }

    // Parse kernel and memcpy records
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* kernel = reinterpret_cast<CUpti_ActivityKernel9_Compat*>(record);

      // Validate before trusting the mirrored layout (see the self-check notes
      // above). Emitting a record that failed these checks would put garbage
      // timestamps into the trace, which is worse than omitting it.
      if (!timestampsPlausible(kernel->start, kernel->end)) {
        reportLayoutMismatch("kernel timestamps implausible");
        continue;
      }
      if (!kernelNamePlausible(kernel->name)) {
        reportLayoutMismatch("kernel name pointer unreadable");
        continue;
      }

      libkineto::GenericTraceActivity activity;
      activity.activityType = libkineto::ActivityType::CONCURRENT_KERNEL;

      // `name` is a const char* field in the record (a pointer into CUPTI's
      // own interned string table), valid for the lifetime of the buffer
      // callback. Copy it into our std::string now.
      if (kernel->name != nullptr) {
        activity.activityName = std::string(kernel->name);
      } else {
        activity.activityName = "kernel";
      }

      activity.startTime = kernel->start;
      activity.endTime = kernel->end;
      activity.device = kernel->deviceId;
      activity.resource = kernel->streamId;
      activity.id = kernel->correlationId;

      g_active_session->activities_.push_back(std::move(activity));

    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* memcpy = reinterpret_cast<CUpti_ActivityMemcpy_Compat*>(record);

      if (!timestampsPlausible(memcpy->start, memcpy->end)) {
        reportLayoutMismatch("memcpy timestamps implausible");
        continue;
      }

      libkineto::GenericTraceActivity activity;
      activity.activityType = libkineto::ActivityType::GPU_MEMCPY;
      activity.activityName = "Memcpy";
      activity.startTime = memcpy->start;
      activity.endTime = memcpy->end;
      activity.device = memcpy->deviceId;
      activity.resource = memcpy->streamId;
      activity.id = memcpy->correlationId;

      g_active_session->activities_.push_back(std::move(activity));
    }
  }

  free(buffer);
}

}  // namespace detail

// ===== FlagosCuptiProfilerSession Implementation =====

void FlagosCuptiProfilerSession::start() {
  auto& shim = CuptiShim::get();
  if (!shim.ok) {
    FLAGOS_CUPTI_LOG("[flagos] CUPTI not available in start()\n");
    return;
  }

  // Publish this session and reset its buffer under the lock, then RELEASE the
  // lock before touching CUPTI. cuptiActivityFlushAll() invokes bufferCompleted
  // synchronously on this same thread, and bufferCompleted also locks
  // g_session_mutex -- holding it across the flush self-deadlocks on the
  // non-recursive mutex (observed as a hang in torch.profiler start_trace).
  {
    std::lock_guard<std::mutex> lock(detail::g_session_mutex);
    detail::g_active_session = this;
    activities_.clear();
  }

  FLAGOS_CUPTI_LOG("[flagos] FlagosCuptiProfilerSession::start() called\n");

  // Callbacks are registered globally at init time; just enable activities here
  CUptiResult res1 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  CUptiResult res2 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  FLAGOS_CUPTI_LOG("[flagos] ActivityEnable results: KERNEL=" << res1 << ", MEMCPY=" << res2 << "\n");

  // Force a flush to kickstart CUPTI activity collection (lock released above).
  CUptiResult res3 = shim.ActivityFlushAll(1);
  FLAGOS_CUPTI_LOG("[flagos] Initial ActivityFlushAll result: " << res3 << "\n");

  status_ = libkineto::TraceStatus::RECORDING;
}

void FlagosCuptiProfilerSession::stop() {
  auto& shim = CuptiShim::get();
  if (!shim.ok) {
    return;
  }

  FLAGOS_CUPTI_LOG("[flagos] FlagosCuptiProfilerSession::stop() called\n");

  shim.ActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  shim.ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);

  // Flush all pending activity records - force flag=0 means wait for completion
  FLAGOS_CUPTI_LOG("[flagos] Flushing CUPTI activities...\n");
  CUptiResult flush_res = shim.ActivityFlushAll(0);
  FLAGOS_CUPTI_LOG("[flagos] ActivityFlushAll result: " << flush_res << "\n");

  std::lock_guard<std::mutex> lock(detail::g_session_mutex);
  detail::g_active_session = nullptr;
  status_ = libkineto::TraceStatus::PROCESSING;

  FLAGOS_CUPTI_LOG("[flagos] Captured " << activities_.size() << " GPU activities\n");
}

void FlagosCuptiProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  for (const auto& activity : activities_) {
    activity.log(logger);
  }
}

std::unique_ptr<libkineto::DeviceInfo> FlagosCuptiProfilerSession::getDeviceInfo() {
  return std::make_unique<libkineto::DeviceInfo>(
      /*id=*/0,
      /*sortIndex=*/0,
      /*name=*/"flagos:GPU",
      /*label=*/"GPU");
}

std::vector<libkineto::ResourceInfo> FlagosCuptiProfilerSession::getResourceInfos() {
  std::vector<libkineto::ResourceInfo> resources;
  // Report up to 32 streams (will only show streams that actually had activity)
  for (int i = 0; i < 32; ++i) {
    resources.emplace_back(
        /*deviceId=*/0,
        /*id=*/i,
        /*sortIndex=*/i,
        /*name=*/std::string("Stream ") + std::to_string(i));
  }
  return resources;
}

std::unique_ptr<libkineto::CpuTraceBuffer> FlagosCuptiProfilerSession::getTraceBuffer() {
  return nullptr;  // We use processTrace instead
}

void FlagosCuptiProfilerSession::pushCorrelationId(uint64_t id) {
  auto& shim = CuptiShim::get();
  if (shim.ok && shim.ActivityPushExternalCorrelationId) {
    shim.ActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    detail::g_correlation_push_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void FlagosCuptiProfilerSession::popCorrelationId() {
  auto& shim = CuptiShim::get();
  if (shim.ok && shim.ActivityPopExternalCorrelationId) {
    uint64_t id;
    shim.ActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
    detail::g_correlation_pop_count.fetch_add(1, std::memory_order_relaxed);
  }
}

// ===== FlagosCuptiProfiler Implementation =====

const std::string& FlagosCuptiProfiler::name() const {
  static const std::string kName = "flagos_cupti";
  return kName;
}

const std::set<libkineto::ActivityType>& FlagosCuptiProfiler::availableActivities() const {
  static const std::set<libkineto::ActivityType> kActivities = {
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::GPU_MEMCPY,
  };
  return kActivities;
}

std::unique_ptr<libkineto::IActivityProfilerSession> FlagosCuptiProfiler::configure(
    const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  FLAGOS_CUPTI_LOG("[flagos] FlagosCuptiProfiler::configure called with " << activityTypes.size() << " activity types\n");
  return std::make_unique<FlagosCuptiProfilerSession>();
}

std::unique_ptr<libkineto::IActivityProfilerSession> FlagosCuptiProfiler::configure(
    int64_t profileStartTime,
    int64_t profileDuration,
    const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  return std::make_unique<FlagosCuptiProfilerSession>();
}

// ===== Registration =====

void registerFlagosCuptiProfiler() {
  libkineto::api().registerProfilerFactory(
      []() { return std::make_unique<FlagosCuptiProfiler>(); });
}

// Static initialization: register profiler if CUPTI is available
namespace {
struct CuptiProfilerRegistrar {
  CuptiProfilerRegistrar() {
    auto& shim = CuptiShim::get();
    if (shim.ok) {
      // Register CUPTI callbacks ONCE at initialization - not per-session.
      // The empirically-verified working sequence (see memory:
      // cupti-must-arm-before-cuda-context) arms CUPTI by calling
      // RegisterCallbacks *and* ActivityEnable together at module-load time,
      // matching a manual ctypes probe that reliably captured kernels. Enabling
      // only in session start() (long after libtorch_cuda has initialized its
      // own CUPTI state) captured nothing. We enable here and keep the enables
      // in start() as a harmless re-assertion.
      FLAGOS_CUPTI_LOG("[flagos] Registering CUPTI activity callbacks...\n");
      CUptiResult res = shim.ActivityRegisterCallbacks(
          detail::bufferRequested, detail::bufferCompleted);
      FLAGOS_CUPTI_LOG("[flagos] ActivityRegisterCallbacks result: " << res << "\n");
      CUptiResult en_k = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
      CUptiResult en_m = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
      FLAGOS_CUPTI_LOG("[flagos] init ActivityEnable KERNEL=" << en_k
                << " MEMCPY=" << en_m << "\n");

      registerFlagosCuptiProfiler();
      FLAGOS_CUPTI_LOG("[flagos] FlagosCuptiProfiler registered with kineto\n");
    } else {
      FLAGOS_CUPTI_LOG("[flagos] CUPTI not available, FlagosCuptiProfiler not registered\n");
    }
  }
};
static CuptiProfilerRegistrar g_registrar;
}  // namespace

}  // namespace flagos
}  // namespace c10

// C API for testing correlation push/pop calls
extern "C" {
__attribute__((visibility("default")))
uint64_t flagos_cupti_get_correlation_push_count() {
  return c10::flagos::detail::g_correlation_push_count.load(std::memory_order_relaxed);
}

__attribute__((visibility("default")))
uint64_t flagos_cupti_get_correlation_pop_count() {
  return c10::flagos::detail::g_correlation_pop_count.load(std::memory_order_relaxed);
}

__attribute__((visibility("default")))
void flagos_cupti_reset_correlation_counters() {
  c10::flagos::detail::g_correlation_push_count.store(0, std::memory_order_relaxed);
  c10::flagos::detail::g_correlation_pop_count.store(0, std::memory_order_relaxed);
}
}
