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

#pragma once

#include <kineto/IActivityProfiler.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Forward declarations for CUPTI callback friend functions
struct CUctx_st;
typedef struct CUctx_st* CUcontext;

namespace c10 {
namespace flagos {

// Forward declare callback for friend declaration
namespace detail {
void bufferCompleted(CUcontext, uint32_t, uint8_t*, size_t, size_t);
}

/**
 * FlagosCuptiProfilerSession: IActivityProfilerSession implementation that
 * uses CUPTI Activity API to collect GPU kernel traces.
 */
class FlagosCuptiProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  void start() override;
  void stop() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;
  std::vector<std::string> errors() override { return {}; }
  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

  friend void detail::bufferCompleted(CUcontext, uint32_t, uint8_t*, size_t, size_t);

 private:
  friend class FlagosCuptiProfiler;
  std::vector<libkineto::GenericTraceActivity> activities_;
};

/**
 * FlagosCuptiProfiler: IActivityProfiler implementation for flagos backend
 * that provides CONCURRENT_KERNEL and GPU_MEMCPY activities via CUPTI.
 */
class FlagosCuptiProfiler : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t profileStartTime,
      int64_t profileDuration,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
};

/**
 * registerFlagosCuptiProfiler: Register the CUPTI profiler with kineto.
 * Called automatically at static initialization time if CUPTI is available.
 */
void registerFlagosCuptiProfiler();

}  // namespace flagos
}  // namespace c10

// C API for testing correlation push/pop calls
extern "C" {
__attribute__((visibility("default"))) uint64_t flagos_cupti_get_correlation_push_count();
__attribute__((visibility("default"))) uint64_t flagos_cupti_get_correlation_pop_count();
__attribute__((visibility("default"))) void flagos_cupti_reset_correlation_counters();
}
