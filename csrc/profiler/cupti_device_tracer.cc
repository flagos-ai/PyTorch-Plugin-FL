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
//
// NVIDIA device tracer: the only file in the profiler that knows what CUPTI is.
// Everything above it consumes DeviceEvent through the DeviceTracer interface,
// so adding a vendor means adding a sibling of this file and nothing else.

#include "device_tracer.h"
#include "cupti_shim.h"

#include <cxxabi.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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

// Prefix of CUpti_ActivityMemset4 up to correlationId. Same shape as the memcpy
// record except the leading discriminator is a 4-byte `value` rather than the
// four 1-byte copy/src/dst/flags fields -- so the payload fields land at the
// same offsets, which is why the same plausibility checks apply unchanged.
struct CUpti_ActivityMemset_Compat {
  CUpti_ActivityKind_t kind;
  uint32_t value;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
};

// CUpti_ActivityAPI, used for CUPTI_ACTIVITY_KIND_RUNTIME (and DRIVER). Copied
// verbatim from the cu12 header; this record has been stable for many CUPTI
// releases (no versioned CUpti_ActivityAPI<N> exists), so it is the least
// layout-risky of the mirrors here.
struct CUpti_ActivityAPI_Compat {
  CUpti_ActivityKind_t kind;
  uint32_t cbid;  // CUpti_CallbackId
  uint64_t start;
  uint64_t end;
  uint32_t processId;
  uint32_t threadId;
  uint32_t correlationId;
  uint32_t returnValue;
};

// CUpti_ActivityExternalCorrelation: maps a CUPTI correlationId to the
// externalId we pushed via cuptiActivityPushExternalCorrelationId (i.e. torch's
// own correlation id). Without this we cannot translate CUPTI's numbering into
// the numbering kineto's getLinkedActivity callback expects.
struct CUpti_ActivityExternalCorrelation_Compat {
  CUpti_ActivityKind_t kind;
  uint32_t externalKind;
  uint64_t externalId;
  uint32_t correlationId;
  uint32_t reserved;
};
#pragma pack(pop)

namespace c10 {
namespace flagos {
namespace profiler {

namespace {

// --- record-layout self-check ------------------------------------------------
//
// The structs above mirror ONE CUPTI version's layout by hand. NVIDIA adds
// fields and publishes a new CUpti_ActivityKernel<N> every few releases, so on a
// different CUPTI the mirror can be wrong -- and the failure mode is silent: we
// decode whatever bytes land at those offsets and emit a trace full of empty
// names and zero durations (exactly the bug this code was fixed for once).
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

// CUPTI interns kernel names as the linker emitted them, i.e. mangled for any
// C++ template kernel (`_ZN2at6native...`). torch's own CUDA path demangles
// before logging, so we do too -- otherwise the flagos timeline is unreadable
// where the CUDA timeline is not. Names that are already plain (`ampere_sgemm_*`
// and other extern-"C"-style kernels) fail to demangle with status -2 and are
// passed through untouched.
std::string demangleName(const char* mangled) {
  if (mangled == nullptr || mangled[0] == '\0') {
    return "kernel";
  }
  int status = -1;
  char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    std::string result(demangled);
    free(demangled);
    return result;
  }
  if (demangled != nullptr) {
    free(demangled);
  }
  return std::string(mangled);
}

}  // namespace

class CuptiDeviceTracer;

namespace {
// Global tracer pointer for the buffer callbacks (CUPTI callbacks are C-style
// and cannot capture context).
CuptiDeviceTracer* g_active_tracer = nullptr;
std::mutex g_tracer_mutex;

// Buffer pool for CUPTI activity records
constexpr size_t kBufferSize = 8 * 1024 * 1024;  // 8MB per buffer
constexpr size_t kBufferAlignment = 8;
}  // namespace

/**
 * CuptiDeviceTracer: NVIDIA implementation of the DeviceTracer interface.
 * Owns every CUPTI detail -- buffer callbacks, activity-record layouts, the
 * layout self-check, and external-correlation resolution -- and hands the layer
 * above a flat vector of vendor-neutral DeviceEvents.
 */
class CuptiDeviceTracer : public DeviceTracer {
 public:
  bool available() const override { return CuptiShim::get().ok; }

  void start() override {
    auto& shim = CuptiShim::get();
    if (!shim.ok) {
      FLAGOS_CUPTI_LOG("[flagos] CUPTI not available in start()\n");
      return;
    }

    // Publish this tracer and reset its buffers under the lock, then RELEASE the
    // lock before touching CUPTI. cuptiActivityFlushAll() invokes bufferCompleted
    // synchronously on this same thread, and bufferCompleted also locks
    // g_tracer_mutex -- holding it across the flush self-deadlocks on the
    // non-recursive mutex (observed as a hang in torch.profiler start_trace).
    {
      std::lock_guard<std::mutex> lock(g_tracer_mutex);
      g_active_tracer = this;
      events_.clear();
      external_correlation_.clear();
    }

    FLAGOS_CUPTI_LOG("[flagos] CuptiDeviceTracer::start() called\n");

    // Callbacks are registered globally at init time; just enable activities here
    CUptiResult res1 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    CUptiResult res2 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    CUptiResult res_ms = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    CUptiResult res_rt = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    CUptiResult res_ec =
        shim.ActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    FLAGOS_CUPTI_LOG("[flagos] ActivityEnable results: KERNEL=" << res1
              << ", MEMCPY=" << res2 << ", MEMSET=" << res_ms
              << ", RUNTIME=" << res_rt
              << ", EXTERNAL_CORRELATION=" << res_ec << "\n");

    // Force a flush to kickstart CUPTI activity collection (lock released above).
    CUptiResult res3 = shim.ActivityFlushAll(1);
    FLAGOS_CUPTI_LOG("[flagos] Initial ActivityFlushAll result: " << res3 << "\n");
  }

  void stop() override {
    auto& shim = CuptiShim::get();
    if (!shim.ok) {
      return;
    }

    FLAGOS_CUPTI_LOG("[flagos] CuptiDeviceTracer::stop() called\n");

    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

    // Flush all pending activity records - force flag=0 means wait for completion
    FLAGOS_CUPTI_LOG("[flagos] Flushing CUPTI activities...\n");
    CUptiResult flush_res = shim.ActivityFlushAll(0);
    FLAGOS_CUPTI_LOG("[flagos] ActivityFlushAll result: " << flush_res << "\n");

    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    g_active_tracer = nullptr;

    FLAGOS_CUPTI_LOG("[flagos] Captured " << events_.size() << " GPU activities\n");
  }

  // Resolution of external correlations happens HERE, not at parse time: CUPTI
  // makes no ordering guarantee between an activity record and the
  // EXTERNAL_CORRELATION record that maps it, and they routinely land in
  // different buffers. Resolving per-record would silently lose every mapping
  // that arrived late -- and losing it is invisible (getLinkedActivity returns
  // nullptr, per-op device time reverts to 0, no diagnostic). drain() runs after
  // stop()'s blocking flush, so by then every record has been seen.
  std::vector<DeviceEvent> drain() override {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    for (auto& ev : events_) {
      ev.external_correlation_id = lookupExternalCorrelation(ev.correlation_id);
    }
    return std::move(events_);
  }

  void pushCorrelation(uint64_t id) override {
    auto& shim = CuptiShim::get();
    if (shim.ok && shim.ActivityPushExternalCorrelationId) {
      shim.ActivityPushExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    }
  }

  void popCorrelation() override {
    auto& shim = CuptiShim::get();
    if (shim.ok && shim.ActivityPopExternalCorrelationId) {
      uint64_t id = 0;
      shim.ActivityPopExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
    }
  }

  // Queried through the already-loaded CUDA runtime rather than linked against:
  // this build is CPU-torch + an external libtorch_cuda.so, so libcudart is in
  // the process but not on our link line. cudaGetDeviceCount does not create a
  // context, so this is safe to call at any point.
  int deviceCount() const override {
    static const int count = [] {
      using GetDeviceCountFn = int (*)(int*);
      auto fn = reinterpret_cast<GetDeviceCountFn>(
          dlsym(RTLD_DEFAULT, "cudaGetDeviceCount"));
      int n = 0;
      if (fn != nullptr && fn(&n) == 0 && n > 0) {
        return n;
      }
      return 1;
    }();
    return count;
  }

  // Called from bufferCompleted with g_tracer_mutex held.
  void processBuffer(uint8_t* buffer, size_t validSize) {
    auto& shim = CuptiShim::get();

    CUpti_Activity* record = nullptr;
    while (true) {
      CUptiResult status = shim.ActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;  // No more records
      }
      if (status != CUPTI_SUCCESS || !record) {
        break;
      }

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

        DeviceEvent ev;
        ev.kind = EventKind::Kernel;
        ev.start_ns = kernel->start;
        ev.end_ns = kernel->end;
        ev.correlation_id = kernel->correlationId;
        ev.device = kernel->deviceId;
        ev.stream = kernel->streamId;
        // `name` is a const char* field in the record (a pointer into CUPTI's
        // own interned string table), valid only for the lifetime of the buffer
        // callback. Copy it into our std::string now.
        ev.name = demangleName(kernel->name);
        ev.metadata["grid"] = "[" + std::to_string(kernel->gridX) + "," +
                              std::to_string(kernel->gridY) + "," +
                              std::to_string(kernel->gridZ) + "]";
        ev.metadata["block"] = "[" + std::to_string(kernel->blockX) + "," +
                               std::to_string(kernel->blockY) + "," +
                               std::to_string(kernel->blockZ) + "]";
        ev.metadata["registers per thread"] =
            std::to_string(kernel->registersPerThread);
        ev.metadata["shared memory"] = std::to_string(
            static_cast<int64_t>(kernel->staticSharedMemory) +
            kernel->dynamicSharedMemory);
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        auto* memcpy_rec = reinterpret_cast<CUpti_ActivityMemcpy_Compat*>(record);

        if (!timestampsPlausible(memcpy_rec->start, memcpy_rec->end)) {
          reportLayoutMismatch("memcpy timestamps implausible");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Memcpy;
        ev.start_ns = memcpy_rec->start;
        ev.end_ns = memcpy_rec->end;
        ev.correlation_id = memcpy_rec->correlationId;
        ev.device = memcpy_rec->deviceId;
        ev.stream = memcpy_rec->streamId;
        ev.name = "Memcpy";
        ev.metadata["bytes"] = std::to_string(memcpy_rec->bytes);
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
        auto* memset_rec = reinterpret_cast<CUpti_ActivityMemset_Compat*>(record);

        if (!timestampsPlausible(memset_rec->start, memset_rec->end)) {
          reportLayoutMismatch("memset timestamps implausible");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Memset;
        ev.start_ns = memset_rec->start;
        ev.end_ns = memset_rec->end;
        ev.correlation_id = memset_rec->correlationId;
        ev.device = memset_rec->deviceId;
        ev.stream = memset_rec->streamId;
        ev.name = "Memset";
        ev.metadata["bytes"] = std::to_string(memset_rec->bytes);
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        // Runtime API calls (cudaLaunchKernel etc.) -- the correlation pivot that
        // lets kineto link a CPU op to the device kernel it launched.
        auto* rt = reinterpret_cast<CUpti_ActivityAPI_Compat*>(record);

        if (!timestampsPlausible(rt->start, rt->end)) {
          reportLayoutMismatch("runtime timestamps implausible");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Runtime;
        ev.start_ns = rt->start;
        ev.end_ns = rt->end;
        ev.correlation_id = rt->correlationId;
        ev.thread_id = rt->threadId;
        // The name MUST come from the cbid. Hardcoding "cudaLaunchKernel" (as an
        // earlier revision did) mislabels every non-launch entry point -- most
        // visibly a 21ms blocking cudaStreamSynchronize showing up as a launch,
        // which reads as a pathological kernel-launch cost rather than a sync.
        ev.name = cuptiRuntimeCbidToName(rt->cbid);
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        // Not emitted as an event; recorded so drain() can translate CUPTI
        // correlation ids into torch correlation ids.
        auto* ext =
            reinterpret_cast<CUpti_ActivityExternalCorrelation_Compat*>(record);

        // This record has no timestamps to sanity-check, but it is the branch
        // whose silent misdecode is *least* visible: a wrong layout yields
        // garbage externalIds, getLinkedActivity then returns nullptr for
        // everything, and per-op device time silently reverts to 0 with no
        // diagnostic at all. externalKind is the one field we can verify --
        // pushCorrelation only ever pushes CUSTOM0, so anything else means we
        // are reading the wrong offsets.
        if (ext->externalKind != CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
          reportLayoutMismatch("external correlation kind unexpected");
          continue;
        }

        external_correlation_[ext->correlationId] = ext->externalId;
      }
    }
  }

 private:
  // Returns the torch correlation id for a CUPTI correlation id, or nullopt when
  // no EXTERNAL_CORRELATION record mapped it (GPU work issued outside a profiled
  // RecordFunction: allocator traffic, warmup, pre-window kernels).
  //
  // This MUST NOT collapse "not found" onto 0: correlation id 0 is meaningful in
  // torch's scheme (autograd/profiler.py treats corr_id == 0 as a *frontend*
  // function event). Returning 0 for a miss would hand id 0 to getLinkedActivity
  // for every unmapped activity, and any kineto that resolves it would silently
  // attribute all of that device time to one wrong CPU op.
  std::optional<int32_t> lookupExternalCorrelation(
      uint32_t cupti_correlation_id) const {
    auto it = external_correlation_.find(cupti_correlation_id);
    if (it == external_correlation_.end()) {
      return std::nullopt;
    }
    return static_cast<int32_t>(it->second);
  }

  std::vector<DeviceEvent> events_;
  // cupti correlationId -> externalId (torch correlation id), populated from
  // EXTERNAL_CORRELATION records. Guarded by g_tracer_mutex (processBuffer
  // writes it; drain reads it after stop()).
  std::unordered_map<uint32_t, uint64_t> external_correlation_;
};

namespace {

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
  std::lock_guard<std::mutex> lock(g_tracer_mutex);
  if (!g_active_tracer || !buffer) {
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

  g_active_tracer->processBuffer(buffer, validSize);
  free(buffer);
}

// Static initialization: arm CUPTI at module load.
struct CuptiTracerInit {
  CuptiTracerInit() {
    auto& shim = CuptiShim::get();
    if (!shim.ok) {
      FLAGOS_CUPTI_LOG("[flagos] CUPTI not available, tracer not armed\n");
      return;
    }

    // Register CUPTI callbacks ONCE at initialization - not per-session.
    // The empirically-verified working sequence (see memory:
    // cupti-must-arm-before-cuda-context) arms CUPTI by calling
    // RegisterCallbacks *and* ActivityEnable together at module-load time,
    // matching a manual ctypes probe that reliably captured kernels. Enabling
    // only in start() (long after libtorch_cuda has initialized its own CUPTI
    // state) captured nothing. We enable here and keep the enables in start()
    // as a harmless re-assertion.
    //
    // Device-side kinds only (KERNEL/MEMCPY/MEMSET). RUNTIME and
    // EXTERNAL_CORRELATION are armed in start() instead, because they are NOT
    // free for non-profiling users: RUNTIME instruments every CUDA runtime API
    // entry/exit on the CPU side, and arming it here cost a MEASURED ~+24% on a
    // launch-bound workload (10.6us -> 12.9us per op, 3 A/B pairs, far outside
    // the ~0.1ms run-to-run spread) for anyone who merely imports torch_fl. The
    // constraint the memory records is that *callback registration* precede the
    // first CUDA context -- which this still satisfies -- not that every
    // activity kind be armed at import.
    FLAGOS_CUPTI_LOG("[flagos] Registering CUPTI activity callbacks...\n");
    CUptiResult res =
        shim.ActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    FLAGOS_CUPTI_LOG("[flagos] ActivityRegisterCallbacks result: " << res << "\n");
    CUptiResult en_k = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    CUptiResult en_m = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    CUptiResult en_s = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    FLAGOS_CUPTI_LOG("[flagos] init ActivityEnable KERNEL=" << en_k
              << " MEMCPY=" << en_m << " MEMSET=" << en_s
              << " (RUNTIME/EXTERNAL_CORRELATION deferred to start())\n");
  }
};
static CuptiTracerInit g_tracer_init;

}  // namespace

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<CuptiDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10
