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

#include <dlfcn.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Forward declarations to avoid including cupti headers directly.
// This keeps CUPTI include paths out of the main build and prevents
// header pollution. Function pointers are dlopen'd at runtime.

// Opaque CUDA types
typedef struct CUctx_st* CUcontext;

// CUPTI result type (matches cupti_result.h)
typedef enum {
  CUPTI_SUCCESS = 0,
  CUPTI_ERROR_INVALID_PARAMETER = 1,
  CUPTI_ERROR_NOT_INITIALIZED = 15,
  CUPTI_ERROR_MAX_LIMIT_REACHED = 28,
  CUPTI_ERROR_INVALID_KIND = 32
} CUptiResult;

// Activity kinds (subset). Values MUST match the cu12 runtime's
// cupti_activity.h (verified against nvidia-cuda-cupti-cu12): note
// CONCURRENT_KERNEL is 10 in cu12, not 9 as an earlier draft assumed. Using
// the wrong value silently enables/matches the wrong kind (0 kernels captured).
typedef enum {
  CUPTI_ACTIVITY_KIND_INVALID = 0,
  CUPTI_ACTIVITY_KIND_MEMCPY = 1,
  CUPTI_ACTIVITY_KIND_MEMSET = 2,
  CUPTI_ACTIVITY_KIND_KERNEL = 3,
  CUPTI_ACTIVITY_KIND_DRIVER = 4,
  CUPTI_ACTIVITY_KIND_RUNTIME = 5,
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
} CUpti_ActivityKind;

// External correlation kinds (minimal subset)
typedef enum {
  CUPTI_EXTERNAL_CORRELATION_KIND_INVALID = 0,
  CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN = 1,
  CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 3
} CUpti_ExternalCorrelationKind;

// Opaque activity record base
struct CUpti_Activity;

// Callback function types (match cupti_activity.h signatures exactly)
typedef void (*CUpti_BuffersCallbackRequestFunc)(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords);

typedef void (*CUpti_BuffersCallbackCompleteFunc)(
    CUcontext context,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize);

namespace c10 {
namespace flagos {

/**
 * CUPTI Activity API dlopen shim.
 * Dynamically loads CUPTI function pointers at runtime to avoid linking
 * against libcupti.so at build time. This allows the CPU-only build to
 * remain clean while still supporting CUPTI profiling when the runtime
 * environment has CUDA available.
 */
struct CuptiShim {
  bool ok = false;

  // CUPTI Activity API function pointers
  CUptiResult (*ActivityEnable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityDisable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityRegisterCallbacks)(
      CUpti_BuffersCallbackRequestFunc,
      CUpti_BuffersCallbackCompleteFunc) = nullptr;
  CUptiResult (*ActivityFlushAll)(uint32_t) = nullptr;
  CUptiResult (*ActivityGetNextRecord)(
      uint8_t*, size_t, CUpti_Activity**) = nullptr;
  CUptiResult (*ActivityGetNumDroppedRecords)(
      CUcontext, uint32_t, size_t*) = nullptr;
  CUptiResult (*ActivityPushExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t) = nullptr;
  CUptiResult (*ActivityPopExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t*) = nullptr;

  static CuptiShim& get() {
    static CuptiShim inst;
    return inst;
  }

  bool available() const { return ok; }

 private:
  CuptiShim() {
    // CRITICAL (see memory: cupti-must-arm-before-cuda-context): CUPTI must be
    // the copy that matches the CUDA *runtime* actually running in this process.
    // Our architecture is CPU-torch + external libtorch_cuda.so built against
    // cu12.8 (NEEDED libcudart.so.12); _preload_cuda_assets() has already
    // dlopen'd the pip nvidia-cuda-cupti-cu12 `libcupti.so.12` into the process.
    // If we instead dlopen a *different* libcupti (e.g. the system CUDA-13.0
    // one), we arm an instance that is not wired to the running cu12.8 runtime,
    // and the buffer callbacks never fire (0 activities captured).
    //
    // So: first bind to whatever libcupti is ALREADY loaded in the process
    // (RTLD_DEFAULT via dlopen(NULL)), then fall back to the cu12 soname, and
    // only then to generic/other-version paths.
    void* handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
    if (handle && !dlsym(handle, "cuptiActivityRegisterCallbacks")) {
      // Nothing CUPTI-shaped in the already-loaded set; fall through to explicit
      // dlopen of a versioned library.
      handle = nullptr;
    }

    if (!handle) {
      // Prefer the runtime-matching cu12 library; keep other versions as a
      // last resort for environments that ship a single system CUPTI.
      const char* candidates[] = {
          "libcupti.so.12",
          "libcupti.so",
          "libcupti.so.13",
          "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcupti.so"
      };
      for (const char* name : candidates) {
        handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
        if (handle) {
          break;
        }
      }
    }

    if (!handle) {
      return; // CUPTI not available, ok remains false
    }

    // Load function pointers using dlsym
#define LOAD_SYM(field, sym) \
    field = reinterpret_cast<decltype(field)>(dlsym(handle, sym))

    LOAD_SYM(ActivityEnable, "cuptiActivityEnable");
    LOAD_SYM(ActivityDisable, "cuptiActivityDisable");
    LOAD_SYM(ActivityRegisterCallbacks, "cuptiActivityRegisterCallbacks");
    LOAD_SYM(ActivityFlushAll, "cuptiActivityFlushAll");
    LOAD_SYM(ActivityGetNextRecord, "cuptiActivityGetNextRecord");
    LOAD_SYM(ActivityGetNumDroppedRecords, "cuptiActivityGetNumDroppedRecords");
    LOAD_SYM(ActivityPushExternalCorrelationId,
             "cuptiActivityPushExternalCorrelationId");
    LOAD_SYM(ActivityPopExternalCorrelationId,
             "cuptiActivityPopExternalCorrelationId");

#undef LOAD_SYM

    if (getenv("FLAGOS_CUPTI_SHIM_DEBUG")) {
      if (ActivityRegisterCallbacks) {
        Dl_info info;
        if (dladdr(reinterpret_cast<void*>(ActivityRegisterCallbacks), &info) &&
            info.dli_fname) {
          fprintf(stderr, "[flagos-cupti-shim] bound cuptiActivityRegisterCallbacks -> %s\n",
                  info.dli_fname);
        }
      } else {
        fprintf(stderr, "[flagos-cupti-shim] cuptiActivityRegisterCallbacks NOT resolved\n");
      }
    }

    // Mark as available if critical functions loaded successfully
    ok = ActivityEnable && ActivityRegisterCallbacks &&
         ActivityFlushAll && ActivityGetNextRecord;
  }
};

}  // namespace flagos
}  // namespace c10
