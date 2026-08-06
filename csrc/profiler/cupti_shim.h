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
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
  CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39
} CUpti_ActivityKind;

// External correlation kinds (minimal subset)
typedef enum {
  CUPTI_EXTERNAL_CORRELATION_KIND_INVALID = 0,
  CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN = 1,
  CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 3
} CUpti_ExternalCorrelationKind;

// Runtime API callback id -> human name, for naming CUPTI_ACTIVITY_KIND_RUNTIME
// records. Values are CUPTI_RUNTIME_TRACE_CBID_* from cupti_runtime_cbid.h,
// cross-checked three ways: the pip cu12 header (nvidia/cuda_cupti/include), the
// system CUDA-13.0 header, and cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API,
// cbid) queried against the live library -- all 22 entries agree. These ids are
// append-only across CUPTI major versions, so a hardcoded subset is safe.
//
// Getting one of these wrong is invisible at runtime: you get a plausible but
// wrong label rather than an error. Before this mapping existed, EVERY runtime
// record was hardcoded to "cudaLaunchKernel", which reported a 21.9ms blocking
// synchronize as a kernel launch.
//
// This is deliberately a static table rather than a cuptiGetCallbackName() call:
// that symbol is not in the shim's dlsym set, it returns the versioned spelling
// (`cudaLaunchKernel_v7000`) rather than the name torch-cuda's trace uses, and
// resolving it per record would put a library call on the hot parse path.
//
// The `_ptsz` ("per-thread default stream") forms are distinct cbids for the
// same API and MUST map to the same name -- a build compiled with
// --default-stream per-thread emits only those, and would otherwise fall
// through to the generic default.
inline const char* cuptiRuntimeCbidToName(uint32_t cbid) {
  switch (cbid) {
    // --- launches ---
    case 211: return "cudaLaunchKernel";                 // _v7000
    case 214: return "cudaLaunchKernel";                 // _ptsz_v7000
    case 269: return "cudaLaunchCooperativeKernel";      // _v9000
    case 270: return "cudaLaunchCooperativeKernel";      // _ptsz_v9000
    case 430: return "cudaLaunchKernelExC";              // _v11060
    case 431: return "cudaLaunchKernelExC";              // _ptsz_v11060
    // --- transfers ---
    case 31:  return "cudaMemcpy";                       // _v3020
    case 41:  return "cudaMemcpyAsync";                  // _v3020
    case 225: return "cudaMemcpyAsync";                  // _ptsz_v7000
    case 49:  return "cudaMemset";                       // _v3020
    case 51:  return "cudaMemsetAsync";                  // _v3020
    case 235: return "cudaMemsetAsync";                  // _ptsz_v7000
    // --- synchronization (the entries most often misread as launches) ---
    case 131: return "cudaStreamSynchronize";            // _v3020
    case 239: return "cudaStreamSynchronize";            // _ptsz_v7000
    case 165: return "cudaDeviceSynchronize";            // _v3020
    case 137: return "cudaEventSynchronize";             // _v3020
    case 135: return "cudaEventRecord";                  // _v3020
    case 242: return "cudaEventRecord";                  // _ptsz_v7000
    case 147: return "cudaStreamWaitEvent";              // _v3020
    case 247: return "cudaStreamWaitEvent";              // _ptsz_v7000
    // --- allocation ---
    case 20:  return "cudaMalloc";                       // _v3020
    case 22:  return "cudaFree";                         // _v3020
    // Unmapped ids keep a generic label rather than a guessed one: the record is
    // still a real runtime call worth showing, and mislabeling it is worse than
    // leaving it unnamed.
    default:  return "cudaRuntime";
  }
}

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
  // Optional: only used for diagnostics, so a failure to resolve it is not fatal.
  CUptiResult (*GetVersion)(uint32_t*) = nullptr;

  // CUPTI API version of the bound library (0 when unknown). Callers use this to
  // report *which* CUPTI they are talking to, so a record-layout mismatch is
  // attributable instead of showing up as an unexplained empty trace.
  uint32_t api_version = 0;
  // Path of the library the symbols actually came from ("" when unknown).
  const char* library_path = "";

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
    // So the ONLY version-independent rule is: bind whatever libcupti is
    // already loaded in this process (RTLD_DEFAULT via dlopen(NULL)), because
    // that copy is by construction the one the running CUDA runtime pulled in.
    // Naming a specific soname or an absolute /usr/local/cuda-<ver> path would
    // just re-create the original bug on any other CUDA version.
    // Escape hatch, checked FIRST so that an explicit path always wins. It has
    // to outrank the already-loaded copy below, because the situation that
    // motivates setting it -- a preloaded CUPTI whose record layout we cannot
    // decode -- is exactly the situation where a preloaded copy exists. An
    // override that only applied when nothing was loaded would be dead in the
    // one case it is advertised for (see reportLayoutMismatch's diagnostic).
    void* handle = nullptr;
    if (const char* override_path = getenv("FLAGOS_CUPTI_LIBRARY")) {
      handle = dlopen(override_path, RTLD_LAZY | RTLD_LOCAL);
      if (!handle) {
        fprintf(stderr,
                "[flagos-cupti-shim] FLAGOS_CUPTI_LIBRARY=%s could not be "
                "loaded: %s\n",
                override_path, dlerror());
      }
    }

    if (!handle) {
      handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
      if (handle && !dlsym(handle, "cuptiActivityRegisterCallbacks")) {
        // Nothing CUPTI-shaped in the already-loaded set; fall through to
        // explicit dlopen of a versioned library.
        handle = nullptr;
      }
    }

    if (!handle) {
      // Nothing preloaded: try sonames from newest to oldest, then the
      // unversioned name (a devel symlink). This list is a *fallback ordering*,
      // not a supported-version list -- an soname absent from it still works via
      // FLAGOS_CUPTI_LIBRARY or by being preloaded, and the ordering only
      // matters when several are installed but none is loaded.
      const char* candidates[] = {
          "libcupti.so.13",
          "libcupti.so.12",
          "libcupti.so",
      };
      for (const char* name : candidates) {
        if (handle) {
          break;
        }
        handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
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
    LOAD_SYM(GetVersion, "cuptiGetVersion");

#undef LOAD_SYM

    // Record which library and which CUPTI API version we ended up bound to.
    // Both are diagnostics only -- nothing branches on the version, since the
    // whole point is to not hardcode version knowledge.
    if (GetVersion) {
      uint32_t v = 0;
      if (GetVersion(&v) == CUPTI_SUCCESS) {
        api_version = v;
      }
    }
    if (ActivityRegisterCallbacks) {
      Dl_info info;
      if (dladdr(reinterpret_cast<void*>(ActivityRegisterCallbacks), &info) &&
          info.dli_fname) {
        library_path = info.dli_fname;
      }
    }

    if (getenv("FLAGOS_CUPTI_SHIM_DEBUG")) {
      if (ActivityRegisterCallbacks) {
        fprintf(stderr,
                "[flagos-cupti-shim] bound cuptiActivityRegisterCallbacks -> %s "
                "(CUPTI API version %u)\n",
                library_path[0] ? library_path : "<unknown>", api_version);
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
