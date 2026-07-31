# torch_fl PrivateUse1 Profiler 支持 — 设计文档

日期: 2026-07-31
分支/worktree: profiler-support
目标: 让 `torch.profiler` 对 flagos 设备提供与 torch-cuda 等价的能力，
既有算子级 device 计时，也有 CUPTI kernel 级时间线（Chrome trace / TensorBoard）。

## 0. 背景与关键事实（已核实）

torch_fl 是 PyTorch PrivateUse1 "flagos" 后端。当前**无任何 profiler / kineto /
record_function 集成**（`csrc/`、`torch_fl/` 全 grep 为空）；`tests/perf/` 里只有
基于 `TorchDispatchMode` 的应用级计时脚本，不走 `torch.profiler`。

已核实的环境事实（conda env `torch-fl-211`, torch 2.11.0+cpu, A100×2 driver 580）：

1. **CPU 层 profiling 免费可用**：`libtorch_cpu.so` 含 394 个 kineto 符号，
   `ProfilerActivity.CPU` 直接可用，RecordFunction 与 dispatch key 无关。
2. **PrivateUse1 profiler 接入点齐全**：`registerPrivateUse1Methods(ProfilerStubs*)`、
   `privateuse1Stubs()`、`pushPRIVATEUSE1CallbacksStub` 都在库里；Python 侧
   `ProfilerActivity.PrivateUse1` + `ProfilerState::KINETO_PRIVATEUSE1_FALLBACK`
   接线完整（`autograd/profiler.py`、`profiler/profiler.py` 中
   `use_device = _get_privateuse1_backend_name()` → "flagos"）。
3. **内置 CUPTI 是空桩**：CPU wheel 用 `-DLIBKINETO_NOCUPTI` 编译。libtorch_cpu 里
   `CuptiActivityApi::*` / `CuptiActivityProfiler::*` 符号虽存在（T），但**无 libcupti
   dlopen 字符串、无真实 `cuptiActivityEnable` 等 API 字符串**，是编译期替换的 no-op 桩。
   外挂 `libtorch_cuda.so` 的 kineto 符号数为 0。→ 内置 CUPTI 通路不可用。
4. **kineto 外部 profiler 注册接口活着**：`libkineto::api()` 与
   `ActivityProfilerProxy::addChildActivityProfiler(unique_ptr<IActivityProfiler>)`
   符号在 libtorch_cpu.so（T）；公共头
   `include/kineto/{IActivityProfiler,libkineto,ActivityProfilerInterface}.h` 全部 ship。
   → 可自写 `IActivityProfiler` 子类、自己 dlopen 系统 libcupti，不碰 libtorch/libkineto。
5. **CUPTI 物理可用**：头 `/usr/local/cuda-13.0/.../include/cupti_activity.h`，
   运行时库系统 `libcupti.so.13`(2025.3.0) 与 pip `nvidia/cuda_cupti/lib/libcupti.so.12` 均在。
6. **flagos event/stream ABI 已就绪**：`include/flagos.h` 的 `EventCreate/EventRecord/
   EventElapsedTime/...` 对 CUDA vendor 就是 `cudaEvent*` 的 1:1 封装
   （`csrc/runtime/accelerator/cuda/stream.cc`）。`GuardImpl`（`csrc/runtime/guard.h`）
   已把这些接进 `DeviceGuardImplInterface` 的 event 方法。
7. **guard stream 是缺口**：`guard.h:58-77` 所有 stream 返回合成 `Stream(UNSAFE,d,0)`，
   `record`（:125）写死记在 `nullptr` null-stream。→ 非默认流算子计时会错乱，
   与 torch-cuda 一致性要求必须修。

## 1. 决策与范围

用户决策（2026-07-31）：
- 目标是**与 torch-cuda 能力完全一致**，包含 **CUPTI kernel 时间线**。
- **A + B 分阶段**：先做算子级 device 计时（Stage A），再叠 CUPTI child profiler（Stage B）。
- CUPTI 接入方式：**kineto 外部 profiler + 自接系统 libcupti**（不重编 libtorch/libkineto）。
- guard stream 实现：**直接用外挂 libtorch_cuda 的 `c10::cuda::CUDAStream`**，
  非 CUDA vendor 用 `#if` 护住、fallback 到现有合成流。
- `mark/rangePush/rangePop`：Stage A **先 no-op**，NVTX 后补。
- 验证：**单测 + 真实模型（qwen3 infer）**。

不做（YAGNI）：不重编 libkineto/libtorch；不动 `tests/perf/` 既有 TorchDispatchMode 脚本；
不做与本目标无关的重构。

## 2. 架构总览

新增独立编译单元 `csrc/profiler/`，不改动 libtorch/libkineto。两阶段共用该目录。

- **Stage A（算子级 device 计时）**：`FlagosProfilerStubs : torch::profiler::impl::ProfilerStubs`，
  静态初始化 `registerPrivateUse1Methods(&stubs)`。用户
  `profile(activities=[CPU, PrivateUse1])` 时 torch 进入 `KINETO_PRIVATEUSE1_FALLBACK`，
  逐 op 调 `record()`/`elapsed()` 得到每算子 device self-time。前置修复 guard stream。
- **Stage B（CUPTI kernel 时间线）**：`FlagosCuptiProfiler : libkineto::IActivityProfiler`，
  在 session 内自 dlopen 系统 libcupti，用 CUPTI Activity API 采 kernel/memcpy 事件，
  经 `registerProfilerFactory` / `addChildActivityProfiler` 注入 kineto，产出可进
  Chrome trace / TensorBoard 的 GPU 时间线。

## 3. Stage A — 组件与数据流

### 3.1 新增 `csrc/profiler/flagos_profiler_stubs.cc`

```cpp
struct FlagosProfilerStubs : torch::profiler::impl::ProfilerStubs {
  void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override;
  float elapsed(const ProfilerVoidEventStub* e1,
                const ProfilerVoidEventStub* e2) const override;
  void mark(const char*) const override {}         // Stage A: no-op
  void rangePush(const char*) const override {}     // NVTX 后补
  void rangePop() const override {}
  bool enabled() const override { return true; }
  void onEachDevice(std::function<void(int)>) const override;
  void synchronize() const override;               // ::DeviceSynchronize()
};
```

静态初始化调用 `torch::profiler::impl::registerPrivateUse1Methods(new FlagosProfilerStubs())`。

### 3.2 数据流

torch profiler 逐 op 调用（语义对齐 torch 的 `CUDAStubs` 实现）：
- `record(device, event, cpu_ns)`：若 `event->get()` 为空，`EventCreateWithFlags(&ev,
  EventEnableTiming)`，把 `Event_t` 包进 `shared_ptr<void>`（deleter 调 `EventDestroy`）
  写回 `*event`；`EventRecord(ev, <当前 CUDA 流>)`；写回 `*cpu_ns = getTime()`、`*device`。
- 汇总时对每对 event 调 `elapsed(e1, e2)`：`EventElapsedTime(&ms, e1, e2)`，返回
  `ms * 1000`（µs）。
- `synchronize()` → `::DeviceSynchronize()`；`onEachDevice` 遍历 `GetDeviceCount()`。

### 3.3 前置修复 `csrc/runtime/guard.h`（一致性核心）

- `getStream/getDefaultStream/exchangeStream/getNewStream`：CUDA vendor 下改为返回携带
  真实 CUDA StreamId 的 `c10::Stream`，经外挂 libtorch_cuda 的
  `c10::cuda::getCurrentCUDAStream(d)` / `CUDAStream` 取值；非 CUDA vendor（`#if`）
  fallback 到现有 `Stream(UNSAFE,d,0)`。
- `record(event, stream, ...)`：把传入 `c10::Stream` 转 `cudaStream_t` 传给
  `EventRecord`，不再写死 `nullptr`。
- 该修改触及所有 op 走的 guard，需回归单测（普通 op / copy_ / 多流）保证不破坏。

## 4. Stage B — CUPTI child profiler

### 4.1 新增 `csrc/profiler/flagos_cupti_profiler.{h,cc}`

```cpp
class FlagosCuptiProfilerSession : public libkineto::IActivityProfilerSession {
  void start() override;   // cuptiActivityEnable(KERNEL|MEMCPY|MEMSET|RUNTIME) + RegisterCallbacks
  void stop() override;    // cuptiActivityFlushAll
  void processTrace(libkineto::ActivityLogger&) override;   // CUPTI buffer → kineto GenericTraceActivity
  std::unique_ptr<DeviceInfo> getDeviceInfo() override;
  std::vector<ResourceInfo> getResourceInfos() override;    // stream → resource
  void pushCorrelationId(uint64_t) override;                // → cuptiActivityPushExternalCorrelationId
  void popCorrelationId() override;
};

class FlagosCuptiProfiler : public libkineto::IActivityProfiler {
  const std::string& name() const override;                 // "flagos_cupti"
  const std::set<ActivityType>& availableActivities() const override; // CONCURRENT_KERNEL, GPU_MEMCPY...
  std::unique_ptr<IActivityProfilerSession> configure(...) override;
};
```

### 4.2 CUPTI 自接（不碰 libtorch 内置桩）

- 运行时 `dlopen("libcupti.so.12"/".so.13")`，`dlsym` 取
  `cuptiActivityEnable/RegisterCallbacks/FlushAll/GetNextRecord/
  ActivityPushExternalCorrelationId` 等。
- 编译期仅从 `/usr/local/cuda-13.0/.../cupti_activity.h` 引入类型/枚举，不链接 cupti。
- 双缓冲 activity buffer（`bufferRequested`/`bufferCompleted` 回调）；`processTrace` 遍历
  `CUpti_ActivityKernel*` / `CUpti_ActivityMemcpy`，转成带 correlationId、device、stream、
  start/end ns 的 kineto `GenericTraceActivity` 喂 logger。

### 4.3 注入 kineto

静态初始化或 `_C._init` 时：
```cpp
libkineto::api().registerProfilerFactory(
    []{ return std::make_unique<FlagosCuptiProfiler>(); });
```
kineto 生成 trace 时回调本 child profiler，将 GPU kernel 时间线并入最终 Chrome trace。

### 4.4 correlation 桥接（可降级验收项）

torch 在 `KINETO_PRIVATEUSE1_FALLBACK` 下 push correlation，session 的
`pushCorrelationId` 转发给 `cuptiActivityPushExternalCorrelationId`，使 kernel 挂到对应 op。
风险：CUPTI external-correlation 与 torch PrivateUse1 fallback 的 correlation 语义能否严丝合缝
是本阶段最不确定处。实现时**先保证 GPU kernel timeline 能出（独立 GPU track）**；
correlation 挂接作为验收项，不达标则降级为"GPU track 存在但不与 op 连线"，correlation 记 follow-up。

## 5. 测试与验证

### Stage A
- `tests/unit/test_profiler_privateuse1.py`：`profile(activities=[CPU, PrivateUse1])` 包住
  matmul/add，断言 `key_averages()` 有 `device_type == PrivateUse1` 条目且
  `self_device_time_total > 0`。
- guard stream 回归：普通 op / `copy_` / `with torch.flagos.stream(s):` 多流场景结果正确；
  `torch.flagos.current_stream()` 的 id 与 guard 记录一致。
- C++ 健全性：profile 一个 op，确认 `record`/`elapsed` 被调且返回正数。

### Stage B
- 单测：profile 后 `prof.export_chrome_trace(path)`，断言导出 JSON 有 `cat == "kernel"`
  （或 GPU track）事件且 count > 0。
- correlation（可降级）：断言部分 GPU kernel 事件带 correlationId 且能对应 CPU op；
  不达标降级为"GPU track 存在"即通过。

### 真实模型
- `tests/integration/test_qwen3_infer.py` 外套 profiler，导出 trace，确认有 op 级 device time（A）
  与 kernel 时间线（B）。

### 构建前置（实现计划第 0 步）
- 本 worktree 当前 `torch_fl._C` 未编，且 FlagGems `libtriton_jit.so` 存在
  `c10::MessageLogger` undefined symbol（pip flag_gems 与 liboperators.so 同源版本问题）。
  实现第一步必须先在本 worktree 编出 `_C`、`import torch_fl` 通过，否则验证无法运行。

### CMake
- `csrc/profiler/` 加入 `csrc/CMakeLists.txt`。Stage A 无新依赖；Stage B 需 cupti include
  路径（仅编译期，运行时 dlopen）。

## 6. 交付顺序

1. Step 0：worktree 编出 `_C`，`import torch_fl` 通过（解决 triton 符号问题）。
2. Stage A：guard stream 修复 + `FlagosProfilerStubs` + 单测（含 guard 回归）。
3. Stage A 验证：qwen3 infer 外套 profiler，确认 op 级 device time。
4. Stage B：`FlagosCuptiProfiler` + CUPTI 自接 + kineto 注入 + 单测。
5. Stage B 验证：Chrome trace 有 GPU kernel track；correlation 作可降级验收项。
