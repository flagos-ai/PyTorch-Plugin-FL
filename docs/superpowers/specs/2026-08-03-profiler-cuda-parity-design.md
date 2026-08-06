# torch_fl profiler 与 torch-cuda 能力对齐 — 设计文档

日期: 2026-08-03
分支/worktree: profiler-support
前置文档: `2026-07-31-privateuse1-profiler-design.md`（Stage A/B 初版设计）

目标: 让 `torch.profiler` 对 flagos 设备产出的 trace 在**结构上**与 torch-cuda 一致 ——
事件类别、op↔kernel 连线、kernel 元数据字段、算子级 device time 归属全部对齐。

## 0. 现状与实测差距

Stage B（CUPTI kernel 时间线）已跑通：`torch.profiler` 能拿到真实 GPU kernel 事件，
qwen3 推理下 2261 个具名 kernel，名字与时长均正确。本文档处理的是它与 torch-cuda 之间
**剩余的结构性差距**。

用同一段代码（5 次 `(x@y).relu()` + `sum`，1024×1024）分别在
`torch-cuda-210`（2.10.0+cu128）和 flagos（2.11.0+cpu + 外挂 libtorch_cuda）下出 trace，
实测差距：

| 指标 | torch-cuda | flagos | 
|---|---|---|
| flow 箭头 `ac2g` | **59** | **0** |
| `cuda_runtime` 事件 | **34** | **0** |
| kernel args 字段数 | **13** | **0** |
| kernel 名 | `ampere_sgemm_128x64_nn` | `_ZN2at6native...`（未 demangle） |
| `gpu_memset` | 10 | 0 |
| `key_averages` device time 归属 | `aten::mm` = 821µs | 只挂 kernel 名下，`aten::*` 全 0 |

### 根因：整条 correlation 链断了

torch-cuda 的链路（实测确认，非推断）：

```
cpu_op (External id=2)
   ↓ 同一 External id
cuda_runtime "cudaLaunchKernel" (correlation=13)   ← 跑在 CPU 线程 (pid=2102704, tid=2102704)
   ↓ 同一 correlation
kernel ampere_sgemm (correlation=13, External id=2) ← 跑在 GPU stream (pid=0, tid=7)

flow: s@runtime(id=13) ──ac2g──> f@kernel(id=13)
```

实测验证：
- `cpu_op ∩ runtime` 的 External id = 15 个，`cpu_op ∩ kernel` = 同样 15 个
- runtime 与 kernel 共享 15 个 correlation id
- flow 起点 id ⊆ runtime 的 correlation 集合（True）

`cuda_runtime` 是**枢纽**：它同时持有 `External id`（连回 cpu_op）和 `correlation`（连到 kernel）。
我们没有采集 RUNTIME 活动，所以上表 6 行差距里有 5 行源于这一处。

### 两个机制性发现

1. **`IActivityProfilerSession::processTrace` 有四参重载**
   （`IActivityProfiler.h:104`），签名含
   `getLinkedActivityCallback = std::function<const ITraceActivity*(int32_t)>`
   —— kineto 用 correlationId 反查 CPU 侧 activity 并交还给我们，这是填 `linked` / `flow`
   的官方通道。**我们只覆写了单参版本**，所以该回调从未被调用。这是 flow=0 的机制性原因。

2. **Stage A 的 `ProfilerStubs` 是结构性死代码**。
   `autograd/profiler.py:330` 是 if/else 二选一：
   `ProfilerActivity.PrivateUse1 in _supported_activities()` 为 True（实测确认）
   → 走 `kineto_activities.add(PrivateUse1)`，即普通 kineto 路径；
   只有为 False 才降级到 `KINETO_PRIVATEUSE1_FALLBACK` 去调 stubs。
   实测 `profiler_kind` = `ProfilerState.KINETO`，且 `privateuse1_elapsed_us()` 全为 0。
   torch-cuda 自己走的也是 kineto 路径。
   → 原测试里 "requires torch+cuda wheel" 的 skip 理由是错的，但"这条路不该走"的结论对。

## 1. 决策（2026-08-03，用户确认）

| 决策点 | 选择 |
|---|---|
| 验收线 | **结构一致 + 自动 diff 门禁**；数值耗时不比对 |
| 基线来源 | **固化快照进仓**，附重生成脚本；CI 不需要 CUDA torch |
| 采集器抽象 | **分层**：通用 kineto 层 + 可插拔 vendor 采集器 |
| metadata 表示 | **窄核心 + 开放 `map<string,string>`** |
| correlation 路线 | **走 RUNTIME 活动**（与 torch-cuda 同机制） |
| Stage A stubs | **删除** |
| activity kinds | KERNEL + MEMCPY + RUNTIME + MEMSET |
| runtime 的 ActivityType | **先试 `PRIVATEUSE1_RUNTIME`，不达标回退 `CUDA_RUNTIME`** |

不做（YAGNI）：不重编 libkineto/libtorch；不采 DRIVER/OVERHEAD/CUDA_SYNC；
不追求 `overhead`、`Activity Buffer Request` 等 kineto 内部记账事件。

### 被否决的方案

- **重编 libkineto 打开 `LIBKINETO_NOCUPTI`**：kineto 静态编进 `libtorch_cpu.so`，
  等于重编 libtorch，与「CPU torch + 外挂 libtorch_cuda」的架构前提冲突，且每次 torch 升级都要重编。
- **复用外挂 `libtorch_cuda.so` 的 CUDA profiler 通路**：该 so 的 kineto 符号数实测为 0，无路可走。

## 2. 架构

```
csrc/profiler/
  device_tracer.h            ← 新增。vendor 无关接口，零 CUPTI 类型
  cupti_device_tracer.cc     ← 新增。NVIDIA 实现（现有采集逻辑迁入）
  flagos_kineto_profiler.cc  ← 现 flagos_cupti_profiler.cc 改名重写。通用层
  cupti_shim.h               ← 保留（已完成版本解耦，见 §6）
  flagos_profiler_stubs.cc   ← 删除
```

### 2.1 vendor 无关接口

```cpp
// device_tracer.h
namespace c10::flagos::profiler {

enum class EventKind { Kernel, Memcpy, Memset, Runtime };

struct DeviceEvent {
  EventKind kind;
  uint64_t start_ns = 0, end_ns = 0;
  uint32_t correlation_id = 0;   // runtime ↔ kernel 的连接钥匙
  uint32_t device = 0, stream = 0;
  uint32_t thread_id = 0;        // Runtime 事件专用：它落在 CPU 线程上
  std::string name;              // 已 demangle
  std::map<std::string, std::string> metadata;  // grid/block/registers/...
};

class DeviceTracer {
 public:
  virtual ~DeviceTracer() = default;
  virtual bool available() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<DeviceEvent> drain() = 0;
  virtual void pushCorrelation(uint64_t) {}
  virtual void popCorrelation() {}
  virtual int deviceCount() const = 0;
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer();  // 编译期按 vendor 选
}
```

`metadata` 用 `map<string,string>` 而非强类型成员：CUPTI 采集器填
`metadata["grid"] = "[8,16,5]"`，通用层无脑
`for (auto& [k,v] : ev.metadata) act.addMetadata(k, v);`。
加厂商、加字段都不动接口；代价是字段名拼写靠约定，编译期不检查。

### 2.2 通用层职责

`flagos_kineto_profiler.cc` 只认 `DeviceEvent`，**一行 CUPTI 代码都没有**：
- 实现 `libkineto::IActivityProfiler` / `IActivityProfilerSession`，注册进 kineto
- 四参 `processTrace` 里建立 `linked` / `flow`（§3）
- 把 `DeviceEvent::metadata` 灌进 `GenericTraceActivity::addMetadata`
- 产出 `DeviceInfo` / `ResourceInfo`

日后接 ascend 只需写 `cann_device_tracer.cc`，通用层不动。

## 3. correlation 链路（方案核心）

### 3.1 实现步骤

1. **采集 RUNTIME 活动**（新增 `CUPTI_ACTIVITY_KIND_RUNTIME`），
   得到带 `correlationId` 的 host 侧事件，落在 CPU 线程（`thread_id` 字段）。
2. **实现四参 `processTrace(logger, getLinkedActivity, startTime, endTime)`**。
3. 对每个事件，用 `getLinkedActivity(correlation_id)` 拿回 CPU 侧 activity，
   填 `activity.linked`。
4. 成对设置 flow：
   - runtime 端：`flow.id = correlation_id`、`flow.type = kLinkAsyncCpuGpu`(=2)、`flow.start = true`
   - kernel 端：同 id/type，`flow.start = false`

### 3.2 device time 归属是免费副产品

torch 的 `autograd/profiler.py:710-735`：

```python
corr_id = kineto_event.linked_correlation_id()
if corr_id > 0:
    device_corr_map[corr_id].append(fe)
...
for fe in frontend_function_events:
    if fe.id in device_corr_map:
        for f_evt in device_corr_map[fe.id]:
            if f_evt.device_type in [DeviceType.CUDA, DeviceType.PrivateUse1, ...]:
                fe.append_kernel(f_evt.name, f_evt.device_index,
                                 f_evt.time_range.end - f_evt.time_range.start)
```

`GenericTraceActivity::correlationId()` 返回 `id` 字段，torch 侧
`linked_correlation_id()` 取的就是这条链接。**只要 `linked` 设对，
`aten::mm` 的 `self_device_time_total` 自动就有 —— 不需要碰 torch 任何代码。**

> 待验证：此判断由 `autograd/profiler.py:710-735` 的 `device_corr_map` 逻辑与
> `IActivityProfiler.h:78` 的 `getLinkedActivityCallback` 签名推出，两处均已读到实际代码，
> 但未端到端实测。实现阶段第一步即验证（§5 Step 1）。

## 4. 采集范围与字段填充

### 4.1 activity kinds

| kind | 状态 | 产出类别 |
|---|---|---|
| `CONCURRENT_KERNEL` | 已有 | `kernel` |
| `MEMCPY` | 已有 | `gpu_memcpy` |
| `RUNTIME` | **新增** | `privateuse1_runtime` / `cuda_runtime` |
| `MEMSET` | **新增** | `gpu_memset` |

不采 DRIVER / OVERHEAD / CUDA_SYNC。

### 4.2 kernel 的 13 个 metadata 字段

对齐 torch-cuda 的 kernel `args`：
`grid`、`block`、`registers per thread`、`shared memory`、`stream`、`context`、
`device`、`correlation`、`External id`、`queued`、`blocks per SM`、`warps per SM`、
`est. achieved occupancy %`。

其中 grid / block / registers / shared memory 已在现有
`CUpti_ActivityKernel9_Compat` 中解出，只是未写进 metadata；
occupancy 三项需由 SM 数与 block 数推导，属本轮增量。

### 4.3 demangle

`abi::__cxa_demangle`，已验证：
- mangled 名 → `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4>(long, at::PhiloxCudaState)`
- 已可读的名（`ampere_sgemm_128x64_nn`）→ status=-2，保留原样

### 4.4 runtime 事件的 ActivityType（带验证门的决策）

kineto 枚举有两个候选：`CUDA_RUNTIME`（→ `cuda_runtime`）和
`PRIVATEUSE1_RUNTIME`（→ `privateuse1_runtime`，官方为自定义后端准备）。
两个字符串在 `libtorch_cpu.so` 中均存在。

**先用 `PRIVATEUSE1_RUNTIME`**（语义准确、跨厂商中立）。实现阶段第一步实测三项：
1. chrome trace 里有 flow 箭头
2. `key_averages` 里 `aten::mm` 有 `self_device_time_total > 0`
3. 类别名为 `privateuse1_runtime`

**任一不达标 → 改用 `CUDA_RUNTIME` 重测。** 验收脚本对
`{"cuda_runtime", "privateuse1_runtime"}` 任一即通过。

风险来源：torch 后处理侧靠 `kineto_event.device_type()` 分类，该映射在 C++ 侧，
`PRIVATEUSE1_RUNTIME` 是否被识别为参与 correlation 归属的 device runtime 需实测。

## 5. 验收门禁

### 5.1 固化基线快照

`tests/data/profiler_cuda_baseline.json`，附重生成脚本
（升级 torch 时手动刷新）：

```json
{
  "categories": ["cpu_op", "cuda_runtime", "kernel", "gpu_memset", "ac2g", "cpu_instant_event"],
  "kernel_arg_keys": ["grid", "block", "correlation", "device", "stream",
                      "registers per thread", "shared memory", "External id",
                      "queued", "context", "blocks per SM", "warps per SM",
                      "est. achieved occupancy %"],
  "flow_cat": "ac2g",
  "runtime_cat_equivalents": ["cuda_runtime", "privateuse1_runtime"],
  "generated_by": "torch 2.10.0+cu128 / A100",
  "note": "数值耗时不比对，只比结构"
}
```

`categories` 记录的是 **torch-cuda 侧的原样事实**（那里必然是 `cuda_runtime`）。
比对时 runtime 那一项按 `runtime_cat_equivalents` 做等价类处理 ——
flagos 侧产出 `privateuse1_runtime` 或 `cuda_runtime` 均视为满足（见 §4.4）。
其余类别名要求逐字相同。

CI 只跑 flagos 侧，对照快照断言。**基线由 `torch-cuda-210` 环境生成**
（该环境存在且 `torch.cuda.is_available()` 为 True）。

### 5.2 断言项

1. categories 集合覆盖基线（runtime 类别名接受两种之一）
2. flow 箭头数 > 0，且 op↔kernel 配对
3. kernel args 13 字段齐全
4. `key_averages()` 中 `aten::mm` 等算子 `self_device_time_total > 0`
5. kernel 名已 demangle（不含 `_ZN` 前缀）

### 5.3 CI 接入（必做）

现有 profiler 测试**一个 pytest marker 都没有**，且 `grep profiler .github/` 为空
—— 与 `test_rng_dispatch.py` 曾经的「文件在但 CI 永远选不中」是同类问题，且更彻底。

新测试必须：
- 带 `main_ops` marker（`.github/configs/cuda.yml` 的 CI 选择器）
- 显式接进 cuda workflow
- 落地后用 CI 日志确认真的执行了（不能只看本地绿）

## 6. 错误处理与退化

| 情况 | 行为 |
|---|---|
| CUPTI 不可用（无 GPU / dlopen 失败） | `MakeDeviceTracer()` 的 tracer `available()` 为 false，不注册进 kineto；CPU 侧 profiling 不受影响 |
| 记录布局不匹配 | 已实现的自检拦截：丢弃该记录 + 一次性诊断，指名绑定的库与 API 版本 |
| `getLinkedActivity` 返回 null | 该事件不设 `linked`/`flow`，仍作为独立 GPU 事件发出，不丢数据 |
| 某版本 kineto 不调四参 `processTrace` | 基类默认转发到单参版本，退化为无 flow，不崩 |

### 已完成的版本解耦（commit `a2296e0`）

CUPTI 绑定优先级：`FLAGOS_CUPTI_LIBRARY` → 进程内已加载的 CUPTI → soname fallback。
override 必须**最先**检查：促使用户设置它的场景正是「预加载的 CUPTI 解不出来」，
若只在无预加载时检查，它在唯一被推荐的场景里恰好是死的。

记录解码前做合理性自检（`end >= start`、start 非零、时长 < 1 小时、name 可读非空）
—— 四条都是任何 CUPTI 版本上正常记录都满足的性质，不含 cu12 特有知识。

已验证：pip cu12 CUPTI（API 26）与系统 CUDA-13（API 130000）两个大版本均 21/21 通过；
阈值强制改 1ns 时全部拒绝、一次诊断、不崩。

## 7. 交付顺序

1. **验证门**：用 `PRIVATEUSE1_RUNTIME` 采 RUNTIME + 四参 `processTrace` 建 flow，
   实测三项指标；不达标回退 `CUDA_RUNTIME`。此步决定后续所有工作的地基。
2. 抽出 `device_tracer.h` 接口，现有 CUPTI 逻辑迁入 `cupti_device_tracer.cc`。
3. 通用层补齐：MEMSET 采集、13 个 metadata 字段、demangle、多 device 的
   DeviceInfo/ResourceInfo。
4. 删除 `flagos_profiler_stubs.cc` 及其 skip 掉的测试。
5. 生成基线快照 + 重生成脚本 + 对照测试（带 `main_ops` marker）。
6. 接进 `.github/configs/cuda.yml`，用 CI 日志确认真的执行。
7. qwen3 真实模型回归：确认 trace 有完整 op↔kernel 连线与算子级 device time。
