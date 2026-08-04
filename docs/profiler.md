# torch_fl profiler 架构：与 torch-cuda 的能力对齐

> 完成日期：2026-08-04
> 实测机器：A100-SXM4-40GB；torch 2.11.0+cpu + 外挂 libtorch_cuda.so（cu128）
> 参照基线：torch 2.10.0+cu128，见 `tests/data/profiler_cuda_baseline.json`
> 测试：`tests/integration/test_profiler_parity.py`，7 项结构断言

`torch.profiler.profile(activities=[CPU, PrivateUse1])` 对 flagos 设备产出的 Chrome trace
在**结构上**与 torch+cuda 一致：

- **flow 箭头**（`ac2g`）把 CPU op 连到它启动的 device kernel
- **device time 归属**：`prof.key_averages()` 给出每个算子的 `self_device_time_total`
- **完整 kernel 元数据**：13 个字段，含 grid/block、occupancy、shared memory、寄存器数
- **runtime 事件**带真实 API 名（由 cbid 解码，不是写死的占位符）
- **memcpy / memset** 与 kernel 并列采集

本文写给两类读者：要接新硬件 profiler 的人，读 §1 三层架构；要改 correlation 相关代码的人，
读 §2 两套 correlation id —— **那是本代码库最容易犯、且不报错的一类 bug**。

---

## 1. 三层架构

整个 Task 2–4 重构的目的只有一个：**加一个厂商 = 写一个文件**。

```
csrc/profiler/
  device_tracer.h              ← 厂商无关接口（DeviceTracer / DeviceEvent / EventKind）
  cupti_device_tracer.cc       ← NVIDIA 实现；所有 CUPTI 类型只出现在这里
  flagos_kineto_profiler.{h,cc}← 通用 kineto 适配层；零厂商耦合
  cupti_shim.h                 ← dlopen 系统 libcupti 的符号绑定（NVIDIA 专用）
```

### 1.1 厂商无关接口 —— `device_tracer.h`

每个厂商需要满足的契约，全部内容如下：

```cpp
enum class EventKind { Kernel, Memcpy, Memset, Runtime };

struct DeviceEvent {
  EventKind kind;
  uint64_t start_ns, end_ns;
  uint32_t correlation_id;                         // CUPTI id：配对 runtime↔kernel
  std::optional<int32_t> external_correlation_id;  // torch id：驱动 device time 归属
  uint32_t device, stream, thread_id;
  std::string name;                                // 已 demangle
  std::map<std::string, std::string> metadata;     // "grid" → "[8,16,5]" 等
};

class DeviceTracer {
  virtual bool available() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<DeviceEvent> drain() = 0;
  virtual void pushCorrelation(uint64_t id) {}
  virtual void popCorrelation() {}
  virtual int deviceCount() const = 0;
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer();   // 工厂
```

`EventKind` / `DeviceEvent` / `DeviceTracer` 是 kineto 适配层**唯一**认识的三个类型。
所有厂商特有的东西（CUPTI activity record、枚举值、结构体 layout 镜像）都在这层之下。

注意 `external_correlation_id` 是 `std::optional` 而不是 `int32_t`：**0 是合法的 torch
correlation id**，用 0 表示"没有"会静默地把 device time 归属到错误的算子上。

### 1.2 NVIDIA 实现 —— `cupti_device_tracer.cc`

所有 CUPTI 类型、cbid 表、activity record layout 镜像只在这个文件里出现：

- `CuptiTracerInit`（文件级 static）—— 模块加载时 arm CUPTI，见 §3.1
- `bufferRequested` / `bufferCompleted` —— CUPTI activity buffer 回调
- `CuptiDeviceTracer::processBuffer()` —— 把 CUPTI activity record 解码成 `DeviceEvent`
- `cuptiActivityPushExternalCorrelationId` / `Pop...` —— correlation 压栈/出栈
- kernel 名 demangle（`abi::__cxa_demangle`）
- 13 个 kernel 元数据字段，与 torch-cuda 完全一致：
  `grid`、`block`、`registers per thread`、`shared memory`、`warps per SM`、
  `blocks per SM`、`est. achieved occupancy %`、`queued`、`context`、`stream`、
  `device`、`correlation`、`External id`
- `MakeDeviceTracer()` 的唯一定义在本文件末尾

### 1.3 通用 kineto 适配层 —— `flagos_kineto_profiler.{h,cc}`

**零厂商耦合**：这个文件不 include 任何 CUPTI 头，也不出现任何厂商特有类型。它只做两件事
—— 把 `DeviceEvent` 翻成 `libkineto::GenericTraceActivity`，以及 correlation / flow 接线。

- `FlagosKinetoProfiler` / `FlagosKinetoProfilerSession` 实现 kineto 的
  `IActivityProfiler` / `IActivityProfilerSession`
- **必须覆写四参 `processTrace`**（`ActivityLogger&`、`getLinkedActivityCallback`、
  `startTime`、`endTime`）。只覆写单参版本的话 kineto 永远不会把 resolver 交给我们，
  flow 恒为 0 —— 这正是本项目开工时的状态。
- **采集窗口过滤**：丢弃 `[startTime, endTime]` 之外的事件（Task 1 finding 5），见 §3.2
- **flow 箭头**：`activity.flow.id = correlation_id`，runtime 事件 `flow.start = 1`、
  device 事件 `flow.start = 0`。两半必须带**同一个 id**（厂商 correlation id）viewer 才画得出。
- **device time 连线**：`activity.linked = getLinkedActivity(*external_correlation_id)`
  —— 键是 **torch** correlation id，不是 CUPTI 的那个。搞混两者见 §2。

### 1.4 加一个新厂商要做什么

1. 写 `csrc/profiler/{vendor}_device_tracer.cc`，实现 `DeviceTracer`，并在其中定义
   `MakeDeviceTracer()`。
2. 让 CMake 对该 `ACCELERATOR` 只编你这一个 tracer。**注意当前状态**：
   `csrc/CMakeLists.txt` 用的是对整个 `csrc/` 的 `GLOB_RECURSE`，所以今天只要两个
   `*_device_tracer.cc` 同时存在，两份 `MakeDeviceTracer()` 定义就会在链接期冲突。
   因此**第二个接入的厂商需要顺带把按厂商选源文件这件事做掉**（`if(ACCELERATOR STREQUAL ...)`
   显式列源文件，或在各 tracer 内部加 `#ifdef` 守卫）。这是厂商分层里唯一"设计了但还没被
   走通"的一环，如实记在这里。
3. 完成。kineto 适配层不需要任何改动。

---

## 2. 两套 correlation id（最重要的一节）

**这是本代码库最常见的 profiler bug，且它不报错。**

trace 里有两套完全独立的编号，名字像、类型像、都叫 "correlation"，但含义不同：

| | `correlation_id` | `external_correlation_id` |
|---|---|---|
| 是谁的编号 | **CUPTI** 的 | **torch** 的 |
| 来源 | CUPTI activity record 自带 | 从 CUPTI `EXTERNAL_CORRELATION` record（kind 39）解析 |
| 配对什么 | runtime 调用 ↔ 它产生的 device kernel | device/runtime 活动 ↔ 发起它的 CPU 算子 |
| 用途 | 画 **flow 箭头** | **device time 归属** |
| trace 里的字段 | `args["correlation"]` | `args["External id"]` |
| 代码里用在哪 | `activity.flow.id` | `getLinkedActivity()` 的入参 |

实测的一对（取自本机 trace）：

```
runtime 'cudaLaunchKernel'         correlation=80  External id=3
kernel  'ampere_sgemm_128x64_nn'   correlation=80  External id=3
flow 两半 id=80: [('s', 'ac2g'), ('f', 'ac2g')]      ← 由 correlation 配对
aten::mm 的 self_device_time_total ← 由 External id 归属
```

### 为什么必须是两套

两者数的是不同的东西，谁也推不出谁。CUPTI id 标识"一次 runtime 调用及其产生的 device 活动"，
torch id 标识"一个 `aten::*` 算子"。一个算子通常发出很多次 CUPTI 可见的调用，所以是
**1 个 torch id 对 N 个 CUPTI id**：实测同一条 parity workload trace 上，某个 `aten::mm`
的 External id 覆盖了 **69** 个不同的 CUPTI correlation id，全 trace 的 fan-out 分布从 1 到 69。
CUPTI 的 `EXTERNAL_CORRELATION` record（kind 39）就是这两套编号之间的桥。

### 传错了会怎样

**把 CUPTI id 传给 `getLinkedActivity()`，不会报任何错，只是 `self_device_time_total`
静默变成 0。** Task 1 用 ablation 证过：只屏蔽 `activity.linked` 一行、其余完全不动，
`aten::mm` 的 device time 从 816µs 掉到 0µs，trace 本身照常生成、kernel 时间线照常正确。

反过来，flow 用 torch id 也一样静默出错：会只产出配不上对的 `f` 半边，viewer 一根箭头也画不出来
（历史上出现过 203 个悬空 `f`，而当时的 `count > 0` 断言照样通过）。

所以 `test_profiler_parity.py` 的第 2 项和第 4 项断言分别盯死这两条链路，且都不是
"大于 0 就算过"：flow 断言要求**每个 id 都成对**，device time 断言要求和 trace 内独立算出的
device 事件时长之和**对得上**。

---

## 3. 实现要点

### 3.1 CUPTI 的 arm 时机约束

`cuptiActivityRegisterCallbacks` **必须在第一个 CUDA context 创建之前**调用。这是实测结论
（memory: `cupti-must-arm-before-cuda-context`）：在已有 context 之后再注册，buffer 回调永远
不会被调用，CUPTI 采到 0 条记录。所以注册放在 `cupti_device_tracer.cc` 的文件级 static
`CuptiTracerInit` 里，在 `import torch_fl` 加载动态库时就执行，早于任何设备操作。

但**具体 arm 哪些 activity kind，分成了两个时刻**：

| 静态初始化时 arm | session `start()` 时 arm |
|---|---|
| `CONCURRENT_KERNEL`、`MEMCPY`、`MEMSET` | `RUNTIME`、`EXTERNAL_CORRELATION` |

这个拆分是**性能实测的结果，不是约束**。`RUNTIME` 会对每次 CUDA runtime API 的进入/退出
插桩，在 import 期就 arm 它，实测让一个 launch-bound 负载**慢 22%：10.56 → 12.88 µs/op**
（3 组 A/B；差值约 5ms，而 run-to-run 抖动只有约 0.1ms，且 arm 后方差还放大了约 15 倍）。
对只是 `import torch_fl`、根本不 profile 的用户来说这是实打实的回归。把这两个 kind 推迟到
`start()` 后，回到 **10.48 µs/op**，落回噪声内，同时采集功能完全不受影响。

memory 记录的约束是"**回调注册**要早于第一个 CUDA context"——推迟 kind 仍然满足它——
而不是"每个 kind 都必须在 import 期 arm"。

**值得记住的坑**：一个 GPU-bound 负载（矩阵乘为主，而非 launch 为主）对这个时机完全不敏感，
两种做法测不出差别。**只测 GPU-bound 负载的话，会错误地判定 import 期 arm 是免费的。**

### 3.2 采集窗口过滤

`flagos_kineto_profiler.cc:266` 的 C++ 谓词是一个**区间相交**判断：

```cpp
auto in_window = [startTime, endTime](const profiler::DeviceEvent& ev) {
  return static_cast<int64_t>(ev.end_ns) >= startTime &&
         static_cast<int64_t>(ev.start_ns) <= endTime;
};
```

没有这个过滤会怎样：tracer 的 device 类 kind 是 import 期就 arm 的，会跨整个进程生命周期
持续记录，所以 trace 里会混进 profile 开始之前的活动 —— 实测 258 个 runtime 事件里有 66 个
结束于第一个 cpu_op 之前，其中一条 250ms 的记录出现在一个 157ms 的 span 里。

parity 测试第 7 项（`test_capture_window_containment`）断言的是更强的**严格包含**
（`ts >= lo && ts + dur <= hi`），即 libkineto 自己 `outOfRange` 那套语义。之所以敢用更强的：
实测 3 次捕获、每次 271 个受管事件，零违规，最紧的余量是起始侧约 1.0–1.7ms、结束侧约 30µs；
结束侧那 30µs 是结构性的（收尾的 `cudaDeviceSynchronize` 必然早于 profiler stop）。

将来若它真的因为窗口**起始**侧的小幅跨界而失败，那是"profiler 启动瞬间确实有活动在飞行中"
的**发现**，应当报告；不要把断言悄悄放宽回相交语义。

### 3.3 时间戳的时钟域

`cuptiGetTimestamp()` 和 kineto 的 `[startTime, endTime]` **都是 UNIX epoch 纳秒**
（实测：`cuptiGetTimestamp` 与 `CLOCK_REALTIME` 相差 2.7µs）。用 MONOTONIC / BOOTTIME 读任一侧
都会差约 1.78e18 ns（约 56 年），窗口过滤会把所有东西滤光。所以 §3.2 的比较是同量纲的：
包含性检查若失败，怀疑逻辑或窗口本身，不要怀疑时钟域。

### 3.4 metadata 的 JSON 引号处理

kineto 的 `GenericTraceActivity::addMetadata` 一律以 `quoted=false` 存值，`metadataJson()`
直接输出 `"key": <raw>`。只要有一个裸标识符（kernel 名、memory kind 标签、`N/A`）走这条路，
产出的就是 `"key": N/A` —— 不是合法 JSON。而 kineto 把所有 activity 拼进**同一个文档**，
所以**一个事件就能让整个 trace 文件 `json.load()` 失败**，破坏力与错误本身完全不成比例。

`metadataValueIsJsonLiteral()` 按文本形态判断能否裸写，其余走 `addMetadataQuoted`。
判断刻意保守：数字被多加引号只是显示成字符串（观感问题），非字面量漏了引号是整个文件报废（致命）。
指数形式必须放行——`memory bandwidth (GB/s)` 用 `%g` 格式化以逐字节对齐 torch-cuda，小量级时
会输出 `8e-05`。

---

## 4. 调试用环境变量

两个都默认关闭。注意它们判断的是**是否设置**、而不是值，所以 `FLAGOS_KINETO_SHIM_DEBUG=0`
一样会打开日志；要关掉就 unset。

- **`FLAGOS_KINETO_SHIM_DEBUG`** —— kineto 适配层（`flagos_kineto_profiler.cc`）的诊断：
  session stop 时 drain 到多少事件；`processTrace` 拿到的窗口、linked/候选计数、被窗口丢弃的条数；
  profiler 注册与 `configure()` 调用。

- **`FLAGOS_CUPTI_SHIM_DEBUG`** —— CUPTI tracer（`cupti_device_tracer.cc`）与 dlopen shim
  （`cupti_shim.h`）的诊断：绑定到了哪个 `libcupti` 及其 API version；回调注册与各 kind 的
  `ActivityEnable` 返回值；buffer 请求/完成与逐条 record 解码。

**刻意不受开关控制的两条警告**：`getLinkedActivity` 回调为空的警告，以及 tracer 的
activity record layout 不匹配警告。两者都罕见但后果严重（前者会让 device time 静默变 0），
而"静默"正是它们难查的原因，所以无条件打印。

---

## 5. parity 测试与基线

**测试**：`tests/integration/test_profiler_parity.py`，7 项断言。全部只断言**结构**，
不断言计数和时长 —— 这是共享 GPU，两者都会 run-to-run 漂移，"恰好 18 根箭头"必然 flaky，
"每根箭头都成对"不会。

| # | 断言 | 内容 |
|---|---|---|
| 1 | `test_category_coverage` | flagos 产出 torch-cuda 的每一个类别（runtime 类别有等价类映射） |
| 2 | `test_flow_arrows_are_paired` | 每个 `s` 半边都有同 id 的 `f` 半边（可渲染，而非仅仅存在） |
| 3 | `test_arg_key_supersets` | 各类别 `args` 键是基线的超集（新增字段不算破坏，字段消失才算） |
| 4 | `test_device_time_attribution` | `aten::mm` 的 `self_device_time_total` 等于它拥有的 device 事件时长之和 |
| 5 | `test_kernel_names_are_demangled` | 没有裸的 C++ mangled 符号（`_ZN...`） |
| 6 | `test_runtime_names_come_from_cbid` | 至少一个 runtime 名超出通用兜底名，证明 cbid 表在生效 |
| 7 | `test_capture_window_containment` | 没有 device/runtime 事件越出采集窗口，见 §3.2 |

**基线**：`tests/data/profiler_cuda_baseline.json`，由 `tests/data/gen_profiler_baseline.py`
在原生 torch+cuda（2.10.0+cu128）上采得。里面只有类别、args 键和已知缺口说明，**没有计数和时长**。

**重新生成基线**：

```bash
conda activate torch-cuda-210      # 必须是真 CUDA 的 torch，不能是 torch-fl-211
python tests/data/gen_profiler_baseline.py
```

脚本不接受参数，自己写回 `tests/data/profiler_cuda_baseline.json`；`torch.cuda.is_available()`
为假时会拒绝运行。**不能在 `torch-fl-211` 下跑**：两个环境 libc10 ABI 不兼容，且在那里 import
torch_fl 会让基线失去意义（基线必须来自原生 torch，而非被测实现本身）。

**torch 升级时必须刷新基线**。另外**生成器与测试的 workload 必须保持一致**
（`gen_profiler_baseline.py::run_traced_ops()` 与测试里的 `_run_traced_ops()`），否则两份 trace
不可比。

**CI 跑法**（7 个用例，A100 上约 2 秒；7 项共用一个 module 级 fixture 只捕获一次）：

```bash
PYTHONPATH=$(pwd) bash scripts/with_cuda_libtorch.sh \
    python -m pytest tests/integration/test_profiler_parity.py -v -m main_ops
```

---

## 6. 已知缺口（如实记录，不粉饰）

### 6.1 不采集 `overhead` 类别

flagos 从不 enable `CUPTI_ACTIVITY_KIND_OVERHEAD`，所以 CUPTI 自报的 profiling 开销
（"Activity Buffer Request"、"Runtime Triggered Module Loading"）不会出现。这些衡量的是
profiling 自身的成本、不是用户负载，缺失不影响对负载的任何测量。已连同理由记在基线 JSON 的
`categories_known_gap` 里，而不是悄悄省略。

### 6.2 flow 配对断言比 torch-cuda 本身更严

parity 测试第 2 项要求**每根** flow 箭头都成对。**torch-cuda 自己并不满足这一条**——
同一 workload、连续 3 次、完全可复现：

```
torch-cuda: ac2g s=26  f=78  paired=False
flagos    : ac2g s=24  f=24  paired=True
```

torch-cuda 多出的 52 个 `f` 半边挂在没有发起侧 `s` 的 `cuda_runtime` 事件上
（`cudaDeviceGetAttribute`、`cudaMalloc`、`cudaOccupancyMaxActiveBlocks` 等）。
flagos 每个 device 事件恰好一根成对箭头。

**所以不要把这条断言描述成"与 torch-cuda 对齐"**：它是一条 flagos 自己的、比基线更强的不变量。
保留它是因为 flagos 确实满足，而退化成悬空半边正是它要防的 bug。

---

## 7. 相关资料

- memory `cupti-must-arm-before-cuda-context` —— 回调注册时机约束的实测过程
- memory `torch-fl-211-env` —— 本分支的环境搭建与构建方式（CPU torch + 外挂 libtorch_cuda.so）
- `docs/superpowers/specs/2026-08-03-profiler-cuda-parity-design.md` —— 设计文档与开工时的实测差距表
