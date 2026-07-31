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

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

// Forward declare CUPTI activity record structures we need.
// These match cupti_activity.h but we avoid including the header to keep
// CUPTI paths out of the build.

struct CUpti_Activity {
  uint8_t kind;
};

// Minimal kernel activity record layout (common fields across CUPTI versions)
struct CUpti_ActivityKernel_Compat {
  uint8_t kind;
  uint8_t pad[7];
  uint64_t start;
  uint64_t end;
  uint64_t completed;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  int32_t correlationId;
  // Name follows at some offset, but we'll access via pointer arithmetic
};

// Minimal memcpy activity record layout
struct CUpti_ActivityMemcpy_Compat {
  uint8_t kind;
  uint8_t pad[7];
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  int32_t correlationId;
};

namespace c10 {
namespace flagos {

namespace detail {

// Global session pointer for buffer callbacks (CUPTI callbacks are C-style,
// cannot capture context).
FlagosCuptiProfilerSession* g_active_session = nullptr;
std::mutex g_session_mutex;

// Buffer pool for CUPTI activity records
constexpr size_t kBufferSize = 8 * 1024 * 1024;  // 8MB per buffer
constexpr size_t kBufferAlignment = 8;

void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  std::cerr << "[flagos] CUPTI bufferRequested callback invoked\n";
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
  std::cerr << "[flagos] CUPTI bufferCompleted callback invoked, validSize=" << validSize << "\n";
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
      auto* kernel = reinterpret_cast<CUpti_ActivityKernel_Compat*>(record);

      libkineto::GenericTraceActivity activity;
      activity.activityType = libkineto::ActivityType::CONCURRENT_KERNEL;

      // Extract kernel name (located at offset ~56 bytes in most CUPTI versions)
      // This is fragile but necessary without including cupti_activity.h
      const char* name_ptr = reinterpret_cast<const char*>(
          reinterpret_cast<uintptr_t>(kernel) + 56);
      activity.activityName = std::string(name_ptr);

      activity.startTime = kernel->start;
      activity.endTime = kernel->end;
      activity.device = kernel->deviceId;
      activity.resource = kernel->streamId;
      activity.id = kernel->correlationId;

      g_active_session->activities_.push_back(std::move(activity));

    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* memcpy = reinterpret_cast<CUpti_ActivityMemcpy_Compat*>(record);

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
    std::cerr << "[flagos] CUPTI not available in start()\n";
    return;
  }

  std::lock_guard<std::mutex> lock(detail::g_session_mutex);
  detail::g_active_session = this;
  activities_.clear();

  std::cerr << "[flagos] FlagosCuptiProfilerSession::start() called\n";

  // Callbacks are registered globally at init time; just enable activities here
  CUptiResult res1 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  CUptiResult res2 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  std::cerr << "[flagos] ActivityEnable results: KERNEL=" << res1 << ", MEMCPY=" << res2 << "\n";

  // Force a flush to kickstart CUPTI activity collection
  CUptiResult res3 = shim.ActivityFlushAll(1);
  std::cerr << "[flagos] Initial ActivityFlushAll result: " << res3 << "\n";

  status_ = libkineto::TraceStatus::RECORDING;
}

void FlagosCuptiProfilerSession::stop() {
  auto& shim = CuptiShim::get();
  if (!shim.ok) {
    return;
  }

  std::cerr << "[flagos] FlagosCuptiProfilerSession::stop() called\n";

  shim.ActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  shim.ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);

  // Flush all pending activity records - force flag=0 means wait for completion
  std::cerr << "[flagos] Flushing CUPTI activities...\n";
  CUptiResult flush_res = shim.ActivityFlushAll(0);
  std::cerr << "[flagos] ActivityFlushAll result: " << flush_res << "\n";

  std::lock_guard<std::mutex> lock(detail::g_session_mutex);
  detail::g_active_session = nullptr;
  status_ = libkineto::TraceStatus::PROCESSING;

  std::cerr << "[flagos] Captured " << activities_.size() << " GPU activities\n";
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
  }
}

void FlagosCuptiProfilerSession::popCorrelationId() {
  auto& shim = CuptiShim::get();
  if (shim.ok && shim.ActivityPopExternalCorrelationId) {
    uint64_t id;
    shim.ActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
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
  std::cerr << "[flagos] FlagosCuptiProfiler::configure called with " << activityTypes.size() << " activity types\n";
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
      // Register CUPTI callbacks ONCE at initialization - not per-session
      std::cerr << "[flagos] Registering CUPTI activity callbacks...\n";
      CUptiResult res = shim.ActivityRegisterCallbacks(
          detail::bufferRequested, detail::bufferCompleted);
      std::cerr << "[flagos] ActivityRegisterCallbacks result: " << res << "\n";

      registerFlagosCuptiProfiler();
      std::cerr << "[flagos] FlagosCuptiProfiler registered with kineto\n";
    } else {
      std::cerr << "[flagos] CUPTI not available, FlagosCuptiProfiler not registered\n";
    }
  }
};
static CuptiProfilerRegistrar g_registrar;
}  // namespace

}  // namespace flagos
}  // namespace c10
