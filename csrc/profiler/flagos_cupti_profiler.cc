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
// Kineto adaptor layer. Everything vendor-specific lives behind DeviceTracer
// (csrc/profiler/device_tracer.h); this file only translates DeviceEvent into
// libkineto::GenericTraceActivity and owns the correlation/flow wiring.

// MUST precede every other include: kineto's GenericTraceActivity.h pulls in
// <fmt/format.h>, and once that is parsed without FMT_HEADER_ONLY the inline
// definitions are gone for the rest of the TU. GenericTraceActivity::addMetadata
// is a template calling fmt::format, and libtorch_cpu.so compiles fmt with
// hidden visibility (it exports no fmt::vformat), so a plugin instantiating
// addMetadata otherwise links against an undefined symbol -- observed as
// `undefined symbol: _ZN3fmt3v127vformat...` at import time. Header-only fmt
// resolves it locally; no other TU in this library includes fmt, so there is no
// ODR conflict.
#define FMT_HEADER_ONLY 1
#include <fmt/format.h>

#include "flagos_cupti_profiler.h"
#include "device_tracer.h"

#include <kineto/ActivityType.h>
#include <kineto/ILoggerObserver.h>
#include <kineto/libkineto.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Diagnostic logging is gated behind FLAGOS_CUPTI_SHIM_DEBUG=1 so that normal
// profiling runs stay quiet.
namespace {
inline bool flagos_cupti_debug() {
  static const bool on = (std::getenv("FLAGOS_CUPTI_SHIM_DEBUG") != nullptr);
  return on;
}
}  // namespace
#define FLAGOS_CUPTI_LOG(expr) \
  do { if (flagos_cupti_debug()) { std::cerr << expr; } } while (0)

namespace c10 {
namespace flagos {

namespace detail {

// Instrumentation counters for correlation push/pop verification
std::atomic<uint64_t> g_correlation_push_count{0};
std::atomic<uint64_t> g_correlation_pop_count{0};

namespace {

libkineto::ActivityType toKinetoActivityType(profiler::EventKind kind) {
  switch (kind) {
    case profiler::EventKind::Kernel:
      return libkineto::ActivityType::CONCURRENT_KERNEL;
    case profiler::EventKind::Memcpy:
      return libkineto::ActivityType::GPU_MEMCPY;
    case profiler::EventKind::Memset:
      return libkineto::ActivityType::GPU_MEMSET;
    case profiler::EventKind::Runtime:
      return libkineto::ActivityType::PRIVATEUSE1_RUNTIME;
  }
  return libkineto::ActivityType::CONCURRENT_KERNEL;
}

// Builds the kineto activity for one device event. Correlation/flow wiring is
// applied by the caller, which is the only part that needs the kineto callback.
libkineto::GenericTraceActivity toActivity(const profiler::DeviceEvent& ev) {
  libkineto::GenericTraceActivity activity;
  activity.activityType = toKinetoActivityType(ev.kind);
  activity.activityName = ev.name;
  activity.startTime = static_cast<int64_t>(ev.start_ns);
  activity.endTime = static_cast<int64_t>(ev.end_ns);
  activity.id = static_cast<int32_t>(ev.correlation_id);

  if (ev.kind == profiler::EventKind::Runtime) {
    // Runtime calls happen on the CPU: device 0, and the "resource" lane is the
    // issuing thread rather than a device stream.
    activity.device = 0;
    activity.resource = static_cast<int32_t>(ev.thread_id);
    activity.threadId = static_cast<int32_t>(ev.thread_id);
  } else {
    activity.device = static_cast<int32_t>(ev.device);
    activity.resource = static_cast<int32_t>(ev.stream);
  }

  for (const auto& [key, value] : ev.metadata) {
    activity.addMetadata(key, value);
  }
  return activity;
}

}  // namespace
}  // namespace detail

// ===== FlagosCuptiProfilerSession Implementation =====

FlagosCuptiProfilerSession::FlagosCuptiProfilerSession()
    : tracer_(profiler::MakeDeviceTracer()) {}

void FlagosCuptiProfilerSession::start() {
  events_.clear();
  if (tracer_ && tracer_->available()) {
    tracer_->start();
  }
  status_ = libkineto::TraceStatus::RECORDING;
}

void FlagosCuptiProfilerSession::stop() {
  if (tracer_ && tracer_->available()) {
    tracer_->stop();
    // Drain here rather than in processTrace: stop() is where the tracer has
    // just flushed, and kineto may call either processTrace overload (or none)
    // afterwards.
    events_ = tracer_->drain();
  }
  status_ = libkineto::TraceStatus::PROCESSING;
  FLAGOS_CUPTI_LOG("[flagos] session stop: drained " << events_.size()
                   << " device events\n");
}

void FlagosCuptiProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  FLAGOS_CUPTI_LOG("[flagos] processTrace(1-arg) called with "
                   << events_.size() << " events\n");
  for (const auto& ev : events_) {
    detail::toActivity(ev).log(logger);
  }
}

void FlagosCuptiProfilerSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t startTime,
    int64_t endTime) {
  FLAGOS_CUPTI_LOG("[flagos] processTrace(4-arg) called with "
                   << events_.size() << " events, window=[" << startTime
                   << "," << endTime << "]\n");

  // Capture-window predicate.
  //
  // UNITS (verified, not assumed): kineto's startTime/endTime are ns since the
  // UNIX epoch, and the tracer's event timestamps are the same domain --
  // cuptiGetTimestamp() was measured 2.7us away from CLOCK_REALTIME, i.e. both
  // are realtime-epoch ns, so these compare directly with no scaling.
  // (CLOCK_MONOTONIC/BOOTTIME are ~1.78e18 ns away, so a boot-relative reading
  // of either side would have been off by ~56 years and filtered everything.)
  auto in_window = [startTime, endTime](const profiler::DeviceEvent& ev) {
    return static_cast<int64_t>(ev.end_ns) >= startTime &&
           static_cast<int64_t>(ev.start_ns) <= endTime;
  };

  // Correlation ids that actually have a device-side activity *that we will
  // emit*. A flow needs both ends: emitting an 's' half for a runtime call that
  // launched nothing (e.g. cudaStreamSynchronize, cudaMalloc) -- or whose kernel
  // fell outside the window -- leaves a dangling arrow, which is not what
  // torch+cuda produces.
  std::set<uint32_t> device_correlations;
  for (const auto& ev : events_) {
    if (ev.kind != profiler::EventKind::Runtime && in_window(ev)) {
      device_correlations.insert(ev.correlation_id);
    }
  }

  size_t linked_count = 0;
  size_t dropped_out_of_window = 0;
  for (const auto& ev : events_) {
    // Drop activities outside the capture window. The tracer keeps recording
    // into its buffers across the whole process lifetime (device activity kinds
    // are armed at import time), so without this the trace carries launches from
    // before profiling started -- measured at 66/258 runtime events ending
    // before the first cpu_op, including a 250ms entry against a 157ms span.
    if (!in_window(ev)) {
      ++dropped_out_of_window;
      continue;
    }

    libkineto::GenericTraceActivity activity = detail::toActivity(ev);

    // Resolve the CPU op that produced this device/runtime activity. The
    // callback is keyed on the *torch* correlation id, which is what
    // pushCorrelationId() published to the tracer; the tracer resolved it into
    // external_correlation_id. An absent mapping means "no CPU op to link to" --
    // we must NOT fall back to id 0, which is a valid torch correlation id
    // (hence the optional; see device_tracer.h).
    if (ev.external_correlation_id.has_value() && getLinkedActivity) {
      if (const auto* cpu_activity =
              getLinkedActivity(*ev.external_correlation_id)) {
        // MEASURED (Task 1 ablation): setting `linked` is what makes torch's
        // _parse_kineto_results attribute device time to the CPU op. With this
        // line removed and everything else identical, aten::mm's
        // self_device_time_total drops from ~816us to 0.
        activity.linked = cpu_activity;
        ++linked_count;
      }
    }

    // Flow arrow ("ac2g"): kineto's chrome writer emits one half per activity --
    // 's' when flowStart is set, 'f' otherwise -- and pairs them by flow id. So
    // BOTH halves have to come from us, and both must carry the SAME id, which
    // is the vendor correlation id (shared by a launch's runtime event and the
    // kernel/memcpy event it produced). Using the torch correlation id here
    // instead would emit only unpaired 'f' halves that no viewer renders.
    if (ev.correlation_id != 0 && device_correlations.count(ev.correlation_id)) {
      activity.flow.id = ev.correlation_id;
      activity.flow.type = libkineto::kLinkAsyncCpuGpu;
      activity.flow.start = (ev.kind == profiler::EventKind::Runtime) ? 1 : 0;
    }
    activity.log(logger);
  }

  FLAGOS_CUPTI_LOG("[flagos] processTrace(4-arg) linked " << linked_count << "/"
                   << events_.size() << " activities to CPU ops, dropped "
                   << dropped_out_of_window << " outside the capture window\n");
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
  if (tracer_ && tracer_->available()) {
    tracer_->pushCorrelation(id);
    detail::g_correlation_push_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void FlagosCuptiProfilerSession::popCorrelationId() {
  if (tracer_ && tracer_->available()) {
    tracer_->popCorrelation();
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
      libkineto::ActivityType::GPU_MEMSET,
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
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

// Static initialization: register the kineto profiler when a device tracer is
// actually usable. The tracer's own static init (cupti_device_tracer.cc) is what
// arms the vendor library before the first device context; this registrar only
// cares whether a tracer exists at all.
namespace {
struct CuptiProfilerRegistrar {
  CuptiProfilerRegistrar() {
    auto tracer = profiler::MakeDeviceTracer();
    if (tracer && tracer->available()) {
      registerFlagosCuptiProfiler();
      FLAGOS_CUPTI_LOG("[flagos] FlagosCuptiProfiler registered with kineto\n");
    } else {
      FLAGOS_CUPTI_LOG("[flagos] no device tracer available, "
                       "FlagosCuptiProfiler not registered\n");
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
