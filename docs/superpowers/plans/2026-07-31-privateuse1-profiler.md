# PrivateUse1 Profiler 支持 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `torch.profiler` 对 flagos(PrivateUse1)设备提供与 torch-cuda 等价的能力：算子级 device 计时(Stage A)+ CUPTI kernel 级时间线(Stage B)。

**Architecture:** Stage A 实现 `torch::profiler::impl::ProfilerStubs` 子类经 `registerPrivateUse1Methods` 注册，走已有 flagos event ABI，并修复 guard 使其携带真实 CUDA stream。Stage B 自写 `libkineto::IActivityProfiler` 子类、自 dlopen 系统 libcupti 采 GPU 事件，经 kineto 外部 profiler 接口注入。不重编 libtorch/libkineto。

**Tech Stack:** C++17, PyTorch 2.11 PrivateUse1, libkineto 外部 profiler 接口, CUPTI Activity API (dlopen), CMake, pytest。

## Global Constraints

- 构建/测试环境: conda env `torch-fl-211`, python 3.12, torch `2.11.0+cpu`(CPU wheel，绝不装 pip CUDA torch)。加载 conda: `source /nfs/lvyufeng/env.sh`。
- 构建命令: `FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation`(g++ only，CUDA 符号运行时从外挂 .so 解析)。
- 所有运行/测试必须经包装器: `bash scripts/with_cuda_libtorch.sh <cmd>`(LD_PRELOAD 注入 libtorch_cuda.so；直接 pytest 会 device init 失败)。
- 后端配置: `FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf`。
- Qwen3 测试需: `HF_HOME=/nfs/lvyufeng/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，模型 `Qwen/Qwen3-0.6B`。
- 外挂 CUDA assets 在 `.libtorch_cuda_assets/`(cu128)。CMake 需 `-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE`(CPU wheel 缺 cuda_cmake_macros.h)。
- 硬件: A100-SXM4-40GB ×8, driver 580, 主机 CUDA 13.0 toolkit(用 cu128 userspace)。
- `csrc/CMakeLists.txt` 用 `GLOB_RECURSE *.cc`，新增 `csrc/profiler/*.cc` 自动纳入编译，无需 add_subdirectory。
- CUPTI: 头 `/usr/local/cuda-13.0/targets/x86_64-linux/include/cupti_activity.h`；运行时库系统 `libcupti.so.13` 或 pip `nvidia/cuda_cupti/lib/libcupti.so.12`。
- 提交粒度: 每个 Task 末尾 commit，遵循 TDD、DRY、YAGNI。

---

## File Structure

- `csrc/profiler/flagos_profiler_stubs.cc` (新建) — Stage A: `FlagosProfilerStubs` + 静态注册。
- `csrc/runtime/guard.h` (修改, :58-77 stream 方法, :119-126 record) — 真实 CUDA stream。
- `csrc/profiler/flagos_cupti_profiler.h` (新建) — Stage B: 类声明。
- `csrc/profiler/flagos_cupti_profiler.cc` (新建) — Stage B: CUPTI dlopen + IActivityProfiler 实现 + kineto 注册。
- `csrc/profiler/cupti_shim.h` (新建) — CUPTI 函数指针 dlopen 封装(隔离 cupti 头，避免污染)。
- `tests/unit/test_profiler_privateuse1.py` (新建) — Stage A/B python 单测。
- `csrc/CMakeLists.txt` (修改，仅 Stage B 加 cupti include 路径)。

---

## Task 0: worktree 编出 `_C`，`import torch_fl` 通过

**Files:**
- Modify: 无源码改动(仅构建环境)

**Interfaces:**
- Consumes: 无
- Produces: 可用的 `torch_fl._C` 扩展，为后续所有验证提供运行基础。

- [ ] **Step 1: 确认外挂 CUDA assets 存在**

Run:
```bash
source /nfs/lvyufeng/env.sh && conda activate torch-fl-211
ls -la /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support/.libtorch_cuda_assets/ 2>/dev/null || \
  ls -la /nfs/lvyufeng/PyTorch-Plugin-FL/.libtorch_cuda_assets/
```
Expected: 看到 libtorch_cuda.so / libc10_cuda.so 等。若 worktree 无此目录，从主 checkout 软链或复制:
```bash
ln -s /nfs/lvyufeng/PyTorch-Plugin-FL/.libtorch_cuda_assets \
  /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support/.libtorch_cuda_assets
```

- [ ] **Step 2: 生成算子代码**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
python scripts/codegen_ops.py
```
Expected: 生成 csrc/aten/generated/*.cc(约 1824 ops)，无报错。

- [ ] **Step 3: 构建 `_C`**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation
```
Expected: 编译成功。注: 关闭 FLAGGEMS 以绕开 §背景的 `libtriton_jit.so` undefined symbol `c10::MessageLogger`(pip flag_gems 与 liboperators.so 同源版本问题)。

- [ ] **Step 4: 验证 import 通过**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -c \
  "import torch_fl, torch; x=torch.randn(8,8,device='flagos'); torch.flagos.synchronize(); print('OK', (x@x).sum().item())"
```
Expected: 打印 `OK <number>`，无 ImportError / device init 错误。

- [ ] **Step 5: 记录基线(无需 commit — 纯环境)**

若前述步骤对构建脚本有任何必要修复(如软链逻辑)，则 `git add` 相关文件并:
```bash
git commit -m "chore: worktree build bootstrap for profiler work"
```
否则跳过。

---

## Task 1: guard.h 携带真实 CUDA stream

**Files:**
- Modify: `csrc/runtime/guard.h:58-77`(stream getters), `:119-126`(record)
- Test: `tests/unit/test_profiler_privateuse1.py`(stream 回归部分)

**Interfaces:**
- Consumes: 外挂 libtorch_cuda 的 `c10::cuda::getCurrentCUDAStream(DeviceIndex)`、`c10::cuda::CUDAStream`；flagos ABI `::EventRecord(Event_t, Stream_t)`。
- Produces: guard 的 `getStream/getDefaultStream/exchangeStream/getNewStream` 返回携带真实 CUDA StreamId 的 `c10::Stream`；`record` 在传入流(而非 nullptr)上记录 event。后续 Task 2 的 `record()` 依赖这一点做正确 stream 归因。

- [ ] **Step 1: 写失败测试(多流下 event 归因)**

在 `tests/unit/test_profiler_privateuse1.py` 追加:
```python
import os, sys
import torch
import torch_fl  # noqa: F401


def test_guard_stream_is_real_not_synthetic():
    """guard 返回的 current stream id 应与 torch.flagos.current_stream 一致，
    而不是恒为 0 的合成流。"""
    dev = torch.device("flagos", 0)
    s = torch.flagos.current_stream()
    # 合成流恒为 0；真实流几乎不可能恒为 0(默认流也有非零 unwrap 在多数场景)
    # 用两条不同流断言 id 不同来证明不是写死的 0
    s2 = torch.flagos.Stream()
    with torch.flagos.stream(s2):
        cur = torch.flagos.current_stream()
    assert cur.stream_id == s2.stream_id
    assert s.stream_id != s2.stream_id or s2.stream_id != 0
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_guard_stream_is_real_not_synthetic -v
```
Expected: FAIL(当前 guard 返回合成 id=0，两条流 id 相同)。

- [ ] **Step 3: 修改 guard.h stream getters**

在 `csrc/runtime/guard.h` 顶部(CUDA vendor 区)加入(用现有 vendor 宏护住，参照文件已有 `#if !defined(USE_ASCEND)...` 模式):
```cpp
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_TSINGMICRO)
#include <c10/cuda/CUDAStream.h>
#endif
```
将 `getStream`(:58)改为:
```cpp
  c10::Stream getStream(c10::Device d) const noexcept override {
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_TSINGMICRO)
    auto s = c10::cuda::getCurrentCUDAStream(d.index());
    return c10::Stream(c10::Stream::UNSAFE, d, s.id());
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }
```
同理修改 `getDefaultStream`(用 `getDefaultCUDAStream(d.index())`)、`getStreamFromGlobalPool`(用 `getStreamFromPool(isHighPriority, d.index())`)、`exchangeStream`(用 `setCurrentCUDAStream(CUDAStream(...))` 后返回旧流)、`getNewStream`(用 `getStreamFromPool`)。非 CUDA vendor 分支保留原合成流。

- [ ] **Step 4: 修改 guard.h record 使用传入流**

将 `record`(:119)改为(把 `c10::Stream` 转 `cudaStream_t`):
```cpp
  void record(void** event, const c10::Stream& stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    if (!*event) {
      ::EventCreate((Event_t*)event);
    }
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_TSINGMICRO)
    auto cs = c10::cuda::CUDAStream(c10::cuda::CUDAStream::UNCHECKED,
                                    c10::Stream(c10::Stream::UNSAFE,
                                                stream.device(), stream.id()));
    ::EventRecord(*(Event_t*)event, (Stream_t)cs.stream());
#else
    ::EventRecord(*(Event_t*)event, nullptr);
#endif
  }
```
(注: `cs.stream()` 返回 `cudaStream_t`，`Stream_t` 是 `struct Stream*`，二者按 flagos ABI 约定互转 — 见 `cuda/stream.cc` 里 `(cudaStream_t)stream` 的既有转换。)

- [ ] **Step 5: 重新构建并运行测试**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_guard_stream_is_real_not_synthetic -v
```
Expected: PASS。

- [ ] **Step 6: guard 回归 — 普通 op / copy 不破坏**

Run(跑既有核心 ops 套件确认 stream 改动无回归):
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/integration/ops/ -m "not flaggems and not flaggems_python" -q
```
Expected: 与基线一致(311 passed 附近)，无新增 fail。

- [ ] **Step 7: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add csrc/runtime/guard.h tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): guard.h carries real CUDA stream for event attribution"
```

---

## Task 2: Stage A — FlagosProfilerStubs

**Files:**
- Create: `csrc/profiler/flagos_profiler_stubs.cc`
- Test: `tests/unit/test_profiler_privateuse1.py`(Stage A 部分)

**Interfaces:**
- Consumes: `torch::profiler::impl::ProfilerStubs`(`torch/csrc/profiler/stubs/base.h`)、`registerPrivateUse1Methods`；flagos ABI `EventCreateWithFlags/EventRecord/EventElapsedTime/EventDestroy/DeviceSynchronize/GetDeviceCount`(`include/flagos.h`)；Task 1 修好的 guard record 语义;`torch::profiler::impl::getTime()`。
- Produces: 进程加载时经静态初始化调用 `registerPrivateUse1Methods(new FlagosProfilerStubs())`，使 `profile(activities=[CPU, PrivateUse1])` 走 `KINETO_PRIVATEUSE1_FALLBACK` 得到 op 级 device self-time。

- [ ] **Step 1: 写失败测试(op 级 device 计时)**

在 `tests/unit/test_profiler_privateuse1.py` 追加:
```python
from torch.profiler import profile, ProfilerActivity


def test_stage_a_privateuse1_device_time():
    x = torch.randn(1024, 1024, device="flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(5):
            y = x @ x
        torch.flagos.synchronize()
    ka = prof.key_averages()
    # 至少一个条目有非零 device self-time
    dev_times = [getattr(e, "self_device_time_total", 0) for e in ka]
    assert any(t > 0 for t in dev_times), f"no device time recorded: max={max(dev_times, default=0)}"
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_a_privateuse1_device_time -v
```
Expected: FAIL(无 ProfilerStubs 注册，device time 全 0)。

- [ ] **Step 3: 实现 flagos_profiler_stubs.cc**

Create `csrc/profiler/flagos_profiler_stubs.cc`:
```cpp
// Copyright 2026 FlagOS Contributors. Apache-2.0.
#include <torch/csrc/profiler/stubs/base.h>
#include <c10/util/Exception.h>
#include <functional>
#include <memory>

#include "flagos.h"  // flagos C ABI: Event_t, EventCreateWithFlags, ...

namespace c10::flagos {
namespace {

using torch::profiler::impl::ProfilerStubs;
using torch::profiler::impl::ProfilerVoidEventStub;

struct FlagosProfilerStubs : public ProfilerStubs {
  void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override {
    if (device) {
      int d = 0;
      ::GetDevice(&d);
      *device = static_cast<c10::DeviceIndex>(d);
    }
    if (cpu_ns) {
      *cpu_ns = torch::profiler::impl::getTime();
    }
    Event_t ev = nullptr;
    ::EventCreateWithFlags(&ev, EventEnableTiming);
    ::EventRecord(ev, nullptr);  // 记在当前(默认)流；多流由 guard 路径覆盖
    *event = std::shared_ptr<void>(ev, [](void* p) {
      if (p) ::EventDestroy((Event_t)p);
    });
  }

  float elapsed(const ProfilerVoidEventStub* event,
                const ProfilerVoidEventStub* event2) const override {
    ::EventSynchronize((Event_t)event2->get());
    float ms = 0.0f;
    ::EventElapsedTime(&ms, (Event_t)event->get(), (Event_t)event2->get());
    return ms * 1000.0f;  // µs
  }

  void mark(const char*) const override {}       // Stage A: no-op (NVTX 后补)
  void rangePush(const char*) const override {}
  void rangePop() const override {}
  bool enabled() const override { return true; }

  void onEachDevice(std::function<void(int)> op) const override {
    int count = 0;
    ::GetDeviceCount(&count);
    for (int i = 0; i < count; ++i) op(i);
  }

  void synchronize() const override { ::DeviceSynchronize(); }

  ~FlagosProfilerStubs() override = default;
};

struct RegisterFlagosStubs {
  RegisterFlagosStubs() {
    static FlagosProfilerStubs stubs;
    torch::profiler::impl::registerPrivateUse1Methods(&stubs);
  }
};
static RegisterFlagosStubs g_register_flagos_stubs;

}  // namespace
}  // namespace c10::flagos
```
注: `record` 参数中 torch 传入的 `flag`/`stream` 由 fallback 路径管理；此处按 `CUDAStubs` 语义在当前流记录。若 profiler 复用同一 `*event`(已非空)，torch 侧不会重复调 record，因此这里始终新建。

- [ ] **Step 4: 重新构建并运行测试**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_a_privateuse1_device_time -v
```
Expected: PASS(有非零 device self-time)。若链接报 `registerPrivateUse1Methods` 未定义，确认 csrc 链接的是 `torch_python_library`(见 csrc/CMakeLists.txt:129)且头路径正确。

- [ ] **Step 5: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add csrc/profiler/flagos_profiler_stubs.cc tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): Stage A FlagosProfilerStubs for op-level device timing"
```

---

## Task 3: Stage A 真实模型验证

**Files:**
- Test: 复用 `tests/integration/test_qwen3_infer.py`(外套 profiler，不改原文件；新增独立脚本断言)

**Interfaces:**
- Consumes: Task 2 的 ProfilerStubs 注册。
- Produces: 证据——真实模型下 op 级 device time 可用。

- [ ] **Step 1: 写验证脚本**

Create `tests/integration/test_profiler_qwen3_infer.py`:
```python
import os
import torch
import torch_fl  # noqa: F401
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"


def test_profiler_over_qwen3_infer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("flagos")
    ids = tok("Hello", return_tensors="pt").input_ids.to("flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        with torch.no_grad():
            model(ids)
        torch.flagos.synchronize()
    ka = prof.key_averages()
    dev = [getattr(e, "self_device_time_total", 0) for e in ka]
    assert any(t > 0 for t in dev), "no device time in qwen3 infer profile"
    print(ka.table(sort_by="self_device_time_total", row_limit=10))
```

- [ ] **Step 2: 运行**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  HF_HOME=/nfs/lvyufeng/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/integration/test_profiler_qwen3_infer.py -v -s
```
Expected: PASS，打印含非零 device time 的算子表。

- [ ] **Step 3: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add tests/integration/test_profiler_qwen3_infer.py
git commit -m "test(profiler): Stage A verification over qwen3 infer"
```

---

## Task 4: Stage B — CUPTI dlopen shim

**Files:**
- Create: `csrc/profiler/cupti_shim.h`
- Test: `tests/unit/test_profiler_privateuse1.py`(cupti 可加载探针)

**Interfaces:**
- Consumes: 系统 `libcupti.so.13` / pip `libcupti.so.12`；cupti 头(仅类型/枚举)。
- Produces: `c10::flagos::CuptiShim` 单例，暴露 `bool available()` 与函数指针 `activityEnable/activityRegisterCallbacks/activityFlushAll/activityGetNextRecord/activityPushExternalCorrelationId/activityPopExternalCorrelationId`。Task 5 消费之。

- [ ] **Step 1: 写失败测试(python 侧探针 — cupti 库能定位)**

在 `tests/unit/test_profiler_privateuse1.py` 追加:
```python
import ctypes
import glob


def test_cupti_library_locatable():
    """确认运行环境能 dlopen 到 libcupti(Stage B 前提)。"""
    candidates = ["libcupti.so.13", "libcupti.so.12", "libcupti.so"]
    candidates += glob.glob("/usr/local/cuda-13.0/targets/*/lib/libcupti.so*")
    candidates += glob.glob(
        os.path.join(os.path.dirname(os.__file__),
                     "../site-packages/nvidia/cuda_cupti/lib/libcupti.so*"))
    loaded = None
    for c in candidates:
        try:
            loaded = ctypes.CDLL(c)
            break
        except OSError:
            continue
    assert loaded is not None, f"cannot dlopen libcupti from {candidates}"
```

- [ ] **Step 2: 运行确认(此测试应直接 PASS — 证明库在；若 FAIL 说明环境缺库需先解决)**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_cupti_library_locatable -v
```
Expected: PASS(库存在)。此步是环境守卫，非 TDD 红。

- [ ] **Step 3: 实现 cupti_shim.h**

Create `csrc/profiler/cupti_shim.h`:
```cpp
// Copyright 2026 FlagOS Contributors. Apache-2.0.
#pragma once
#include <cupti_activity.h>  // 仅类型/枚举，运行时符号 dlopen
#include <dlfcn.h>
#include <cstdio>

namespace c10::flagos {

struct CuptiShim {
  bool ok = false;
  CUptiResult (*ActivityEnable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityDisable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityRegisterCallbacks)(
      CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc) = nullptr;
  CUptiResult (*ActivityFlushAll)(uint32_t) = nullptr;
  CUptiResult (*ActivityGetNextRecord)(uint8_t*, size_t, CUpti_Activity**) = nullptr;
  CUptiResult (*ActivityGetNumDroppedRecords)(CUcontext, uint32_t, size_t*) = nullptr;
  CUptiResult (*ActivityPushExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t) = nullptr;
  CUptiResult (*ActivityPopExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t*) = nullptr;

  static CuptiShim& get() {
    static CuptiShim inst;
    return inst;
  }

 private:
  CuptiShim() {
    const char* names[] = {"libcupti.so.13", "libcupti.so.12", "libcupti.so"};
    void* h = nullptr;
    for (auto n : names) {
      h = dlopen(n, RTLD_LAZY | RTLD_GLOBAL);
      if (h) break;
    }
    if (!h) return;
#define LOAD(field, sym) field = (decltype(field))dlsym(h, sym)
    LOAD(ActivityEnable, "cuptiActivityEnable");
    LOAD(ActivityDisable, "cuptiActivityDisable");
    LOAD(ActivityRegisterCallbacks, "cuptiActivityRegisterCallbacks");
    LOAD(ActivityFlushAll, "cuptiActivityFlushAll");
    LOAD(ActivityGetNextRecord, "cuptiActivityGetNextRecord");
    LOAD(ActivityGetNumDroppedRecords, "cuptiActivityGetNumDroppedRecords");
    LOAD(ActivityPushExternalCorrelationId, "cuptiActivityPushExternalCorrelationId");
    LOAD(ActivityPopExternalCorrelationId, "cuptiActivityPopExternalCorrelationId");
#undef LOAD
    ok = ActivityEnable && ActivityRegisterCallbacks && ActivityFlushAll &&
         ActivityGetNextRecord;
  }
};

}  // namespace c10::flagos
```

- [ ] **Step 4: 加 cupti include 路径到 CMake**

Modify `csrc/CMakeLists.txt`(在 `add_library(${LIBRARY_NAME} ...)` 之后)追加:
```cmake
# CUPTI headers for the profiler child (Stage B). Runtime symbols are dlopen'd,
# so we only need the include path, not the link library.
find_path(CUPTI_INCLUDE_DIR cupti_activity.h
  PATHS /usr/local/cuda-13.0/targets/x86_64-linux/include
        /usr/local/cuda/extras/CUPTI/include)
if(CUPTI_INCLUDE_DIR)
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUPTI_INCLUDE_DIR})
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_HAVE_CUPTI=1)
endif()
```

- [ ] **Step 5: 重新构建确认 shim 编译通过**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation
```
Expected: 编译成功(shim 仅头，暂无 .cc 引用它，靠 Task 5 引入；此步只验证 CMake include 生效——可临时在一个既有 .cc 加 `#include "profiler/cupti_shim.h"` 冒烟后回退，或直接进 Task 5)。

- [ ] **Step 6: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add csrc/profiler/cupti_shim.h csrc/CMakeLists.txt tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): Stage B CUPTI dlopen shim + cmake include path"
```

---

## Task 5: Stage B — FlagosCuptiProfiler + kineto 注册

**Files:**
- Create: `csrc/profiler/flagos_cupti_profiler.h`, `csrc/profiler/flagos_cupti_profiler.cc`
- Test: `tests/unit/test_profiler_privateuse1.py`(Chrome trace 有 kernel 事件)

**Interfaces:**
- Consumes: Task 4 的 `CuptiShim`；kineto `libkineto::IActivityProfiler` / `IActivityProfilerSession` / `GenericTraceActivity` / `libkineto::api().registerProfilerFactory`(`torch/include/kineto/*.h`)。
- Produces: 进程加载时经静态初始化 `libkineto::api().registerProfilerFactory(...)` 注册 `FlagosCuptiProfiler`(name `"flagos_cupti"`, activities `CONCURRENT_KERNEL`/`GPU_MEMCPY`)。profile 后 `export_chrome_trace` 含 GPU kernel 事件。

- [ ] **Step 1: 写失败测试(chrome trace 有 kernel 事件)**

在 `tests/unit/test_profiler_privateuse1.py` 追加:
```python
import json
import tempfile


def test_stage_b_chrome_trace_has_gpu_kernels():
    x = torch.randn(1024, 1024, device="flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(5):
            y = x @ x
        torch.flagos.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    kernel_like = [e for e in events
                   if isinstance(e, dict) and (
                       e.get("cat") in ("kernel", "Kernel", "gpu_op") or
                       "kernel" in str(e.get("name", "")).lower())]
    assert len(kernel_like) > 0, "no GPU kernel events in chrome trace"
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_b_chrome_trace_has_gpu_kernels -v
```
Expected: FAIL(无 CUPTI child profiler，trace 无 kernel 事件)。

- [ ] **Step 3: 声明 flagos_cupti_profiler.h**

Create `csrc/profiler/flagos_cupti_profiler.h`:
```cpp
// Copyright 2026 FlagOS Contributors. Apache-2.0.
#pragma once
#include <kineto/IActivityProfiler.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace c10::flagos {

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

 private:
  std::vector<libkineto::GenericTraceActivity> activities_;
};

class FlagosCuptiProfiler : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t profileStartTime, int64_t profileDuration,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
};

void registerFlagosCuptiProfiler();

}  // namespace c10::flagos
```
注: 各虚函数签名必须以本 env 的 `torch/include/kineto/IActivityProfiler.h` 为准 — 实现前先 `grep -n "virtual" torch/include/kineto/IActivityProfiler.h` 逐一核对并对齐(该文件 §设计已确认存在)。

- [ ] **Step 4: 实现 flagos_cupti_profiler.cc**

Create `csrc/profiler/flagos_cupti_profiler.cc`。核心逻辑:
1. `start()`: `CuptiShim::get()` 若 `ok`，`ActivityRegisterCallbacks(bufferRequested, bufferCompleted)` + `ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)`、`ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)`。
2. buffer 回调: `bufferRequested` 分配对齐 buffer；`bufferCompleted` 用 `ActivityGetNextRecord` 遍历，把 `CUpti_ActivityKernel*`/`CUpti_ActivityMemcpy` 转成 `libkineto::GenericTraceActivity`(填 name、device、resource=streamId、startTime=start ns、endTime=end ns、id=correlationId、activityType)，push 进当前 session 的 `activities_`(经全局指针指向活跃 session)。
3. `stop()`: `ActivityFlushAll(1)`。
4. `processTrace(logger)`: 对 `activities_` 逐个 `logger.handleGenericActivity(a)`。
5. `getDeviceInfo`/`getResourceInfos`: 返回设备与 stream 资源描述(name "flagos:GPU")。
6. `pushCorrelationId(id)`: 若 shim 有 `ActivityPushExternalCorrelationId`，`ActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id)`;`popCorrelationId` 对应 pop。
7. `configure(...)` 两个重载都返回 `std::make_unique<FlagosCuptiProfilerSession>()`。
8. `name()` 返回静态 `"flagos_cupti"`；`availableActivities()` 返回静态 `{CONCURRENT_KERNEL, GPU_MEMCPY}`。
9. `registerFlagosCuptiProfiler()`: `libkineto::api().registerProfilerFactory([]{ return std::make_unique<FlagosCuptiProfiler>(); });`
10. 文件底部静态初始化: `struct R { R(){ if (CuptiShim::get().ok) registerFlagosCuptiProfiler(); } } g_r;`

(完整实现体依 kineto 头签名填充；GenericTraceActivity 字段名以 `torch/include/kineto/output_base.h` / `ITraceActivity.h` 为准，实现前 grep 核对。)

- [ ] **Step 5: 重新构建并运行测试**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_b_chrome_trace_has_gpu_kernels -v
```
Expected: PASS(trace 含 GPU kernel 事件)。若 kineto 未回调本 profiler，检查 `registerProfilerFactory` 是否在 `import torch_fl` 时执行(静态初始化在 libtorch_fl.so 加载时触发)。

- [ ] **Step 6: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add csrc/profiler/flagos_cupti_profiler.h csrc/profiler/flagos_cupti_profiler.cc tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): Stage B CUPTI child profiler injected via kineto"
```

---

## Task 6: Stage B correlation 桥接(可降级验收)

**Files:**
- Modify: `csrc/profiler/flagos_cupti_profiler.cc`(correlation 转发已在 Task 5 埋入，此处验证/加固)
- Test: `tests/unit/test_profiler_privateuse1.py`(correlation 断言，可降级)

**Interfaces:**
- Consumes: Task 5 的 session `pushCorrelationId/popCorrelationId`。
- Produces: GPU kernel 事件带 correlationId 且可对应 CPU op(达标)；否则降级为"GPU track 存在"(Task 5 已保证)即视为完成，correlation 记 follow-up。

- [ ] **Step 1: 写 correlation 测试(允许降级)**

在 `tests/unit/test_profiler_privateuse1.py` 追加:
```python
def test_stage_b_correlation_or_degrade():
    x = torch.randn(512, 512, device="flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        y = x @ x
        torch.flagos.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    flows = [e for e in events if isinstance(e, dict) and e.get("ph") in ("s", "t", "f")]
    kernels = [e for e in events if isinstance(e, dict)
               and "kernel" in str(e.get("name", "")).lower()]
    # 达标: 有 flow 事件连接 op↔kernel；降级: 至少 kernel track 存在
    if flows:
        print(f"correlation OK: {len(flows)} flow events")
    else:
        assert len(kernels) > 0, "neither correlation flows nor kernel track present"
        print("DEGRADED: kernel track present, no op<->kernel correlation (follow-up)")
```

- [ ] **Step 2: 运行**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_b_correlation_or_degrade -v -s
```
Expected: PASS(打印 correlation OK 或 DEGRADED)。若 DEGRADED，在 §设计 §4.4 记 follow-up，不阻塞。

- [ ] **Step 3: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add tests/unit/test_profiler_privateuse1.py
git commit -m "test(profiler): Stage B correlation acceptance (degradable)"
```

---

## Task 7: Stage B 真实模型验证 + 收尾

**Files:**
- Test: `tests/integration/test_profiler_qwen3_infer.py`(扩展断言 kernel track)

**Interfaces:**
- Consumes: Task 3 脚本 + Task 5 profiler。
- Produces: 真实模型下 Chrome trace 同时含 op device time 与 GPU kernel 时间线的最终证据。

- [ ] **Step 1: 扩展 qwen3 验证脚本导出 trace 并断言 kernel**

在 `tests/integration/test_profiler_qwen3_infer.py` 追加函数:
```python
import json, tempfile
from torch.profiler import profile, ProfilerActivity


def test_profiler_qwen3_chrome_trace_kernels():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("flagos")
    ids = tok("Hello world", return_tensors="pt").input_ids.to("flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        with torch.no_grad():
            model(ids)
        torch.flagos.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    kernels = [e for e in events if "kernel" in str(e.get("name", "")).lower()]
    assert len(kernels) > 0, "no GPU kernels in qwen3 chrome trace"
    print(f"qwen3 trace: {len(kernels)} kernel events, saved {path}")
```

- [ ] **Step 2: 运行**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  HF_HOME=/nfs/lvyufeng/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/integration/test_profiler_qwen3_infer.py -v -s
```
Expected: 两个测试都 PASS。

- [ ] **Step 3: 全量回归(确保整套改动无副作用)**

Run:
```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/with_cuda_libtorch.sh python -m pytest \
  tests/integration/ops/ -m "not flaggems and not flaggems_python" -q
```
Expected: 与基线一致，无回归。

- [ ] **Step 4: Commit**

```bash
cd /nfs/lvyufeng/PyTorch-Plugin-FL/.claude/worktrees/profiler-support
git add tests/integration/test_profiler_qwen3_infer.py
git commit -m "test(profiler): Stage B verification over qwen3 infer (kernel timeline)"
```

---

## Self-Review Notes

- **Spec 覆盖**: §2 架构→Task 1/2(A)、Task 4/5(B);§3.3 guard→Task 1;§4.2 CUPTI 自接→Task 4;§4.3 kineto 注入→Task 5;§4.4 correlation 可降级→Task 6;§5 测试→Task 3/7 单测+真实模型;§5 构建前置→Task 0;§5 CMake→Task 4 Step 4。全覆盖。
- **类型一致性**: `FlagosProfilerStubs`(Task 2)、`CuptiShim`(Task 4)、`FlagosCuptiProfiler`/`FlagosCuptiProfilerSession`(Task 5/6/7)命名跨任务一致;flagos ABI 函数名以 `include/flagos.h` 为准。
- **已知不确定点**: Task 5 kineto 虚函数签名、GenericTraceActivity 字段名必须实现前对齐本 env 头文件(计划已在步骤内注明 grep 核对);Task 6 correlation 允许降级。
```
