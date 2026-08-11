# 实测：CPU torch + 外挂 libtorch_npu.so 能否复用 Ascend 算子

> 实测日期：2026-07-20
> 实测机器：Ascend 910（8× 910，npu-smi 25.5.0，CANN 9.0.0，aarch64）
> 结论：**不成立（此路径无法直接照搬 CUDA 方案）**。torch_npu 的算子注册在 **PrivateUse1** dispatch key 下，与 torch_fl 的 `flagos`(PrivateUse1) 后端**同键冲突**，不能像 `libtorch_cuda.so`（注册在独立的 `CUDA` key）那样"外挂即兜底"。

## 背景

CUDA 方案（见 [../cuda/external-libtorch-cuda.md](../cuda/external-libtorch-cuda.md)）成立的**根本前提**是：

- PyTorch 的 CUDA kernel 注册在**专属的 `CUDA` dispatch key** 下。
- torch_fl 的 `vm`/`flagos` 后端占用的是 **`PrivateUse1`** key。
- 两者 key 互不重叠 → 外挂 `libtorch_cuda.so` 把 CUDA kernel 塞进 dispatcher 后，boxing 路径把 `flagos` 张量改成 CUDA 元数据再调 `structured_*_out_cuda`，天然分层、不打架。

问题：Ascend 能否照搬——从 torch_npu wheel 抽 `libtorch_npu.so` 外挂，让 NPU kernel 进 dispatcher 供 boxing 兜底？

## 实测步骤

```bash
# 1. 干净 conda 环境 + 只装 CPU torch（版本对齐 torch_npu）
conda create -n libtorch_npu_test python=3.10
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
#   torch/lib 下只有 libc10/libtorch/libtorch_cpu/libtorch_python/libshm/libtorch_global_deps

# 2. 只下载（不安装）版本匹配的 torch_npu wheel，抽出 .so
pip download torch_npu==2.7.1 -d /tmp/npu_wheel --no-deps
#   torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl (22.6 MB)
#   关键产物：torch_npu/lib/libtorch_npu.so (≈51 MB) —— 对标 libtorch_cuda.so

# 3. CANN runtime
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 关键实测结果

### 符号解析 ✅（能加载）

`libtorch_npu.so` 的 `NEEDED` 依赖里含 `libtorch.so / libtorch_cpu.so / libc10.so / libtorch_python.so` + CANN 侧 `libhccl / libascendcl / libge_runner / libgraph` 等。在 `LD_LIBRARY_PATH` 补上 CANN 库路径后，`ctypes.CDLL(libtorch_npu.so, RTLD_GLOBAL)` **加载成功**，无 undefined symbol —— 这一点和 CUDA 一样，CPU wheel 符号喂得饱。

> 注意：CUDA 方案有"必须在 import torch 之前 LD_PRELOAD"的硬约束（CUDAHooks 缓存桩问题）。NPU 实测中在 `import torch` **之后**加载也能把 kernel 注册进表（见下），但设备初始化走的是 `PrivateUse1HooksInterface`——这条 hooks 路径同样会与 torch_fl 自己注册的 hooks 冲突。

### kernel 注册 ❌（同键冲突，这是决定性差异）

```python
import ctypes, torch
def has(op, key): return torch._C._dispatch_has_kernel_for_dispatch_key(op, key)

# 加载前
#   aten::mm         PrivateUse1=False  CPU=True
#   aten::add.Tensor PrivateUse1=False  CPU=True
ctypes.CDLL(".../libtorch_npu.so", ctypes.RTLD_GLOBAL)
# 加载后
#   aten::mm         PrivateUse1=True     ← 注册进了 PrivateUse1！
#   aten::add.Tensor PrivateUse1=True
```

`libtorch_npu.so` 内 `strings` 统计：`PrivateUse1` 出现 25 次、`CUDA` 仅 6 次；并含
`c10_npu::impl::rename_privateuse1_backend()`、`at::RegisterPrivateUse1HooksInterface`、
`c10::register_privateuse1_backend(...)` 等符号。**torch_npu 的整套设备/算子/Hooks 都建立在 PrivateUse1 之上**（这也是社区共识：torch_npu 是 PrivateUse1 out-of-tree backend）。

而 torch_fl 在 `torch_fl/__init__.py` 里正是：

```python
torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", flagos)
```

**同一个 PrivateUse1 key 只能被一个后端占用**。外挂 `libtorch_npu.so` 会：
1. 把 NPU kernel 覆盖/抢注到 `PrivateUse1`，与 flagos 自己的 PrivateUse1 注册互相覆盖；
2. `register_privateuse1_backend` / `PrivateUse1HooksInterface` 与 flagos 的重名冲突（PyTorch 对 PrivateUse1 backend 名与 hooks 只允许注册一次）。

即：**没有"独立 key 分层"这个前提**，boxing 的"改元数据 → 调 native kernel"模型在 NPU 上失去落脚点——目标 key 就是自己占着的那个。

## CUDA vs Ascend 对比

| 维度 | CUDA (`libtorch_cuda.so`) | Ascend (`libtorch_npu.so`) |
|---|---|---|
| kernel 注册 key | **`CUDA`**（独立） | **`PrivateUse1`**（与 flagos 撞） |
| 与 flagos(PrivateUse1) 关系 | 正交，可分层 boxing | 同键，直接冲突 |
| 从 CPU wheel 抽 so 加载 | ✅ 符号完整 | ✅ 符号完整 |
| 设备 Hooks | `CUDAHooks`（需 preload 解决缓存） | `PrivateUse1HooksInterface`（与 flagos hooks 冲突） |
| "外挂即兜底"是否成立 | ✅ 成立 | ❌ 不成立 |

## 结论与建议

- **不能直接照搬 CUDA 方案**。CUDA 之所以成立，靠的是"CUDA key 与 PrivateUse1 key 天然分层"；torch_npu 恰恰把自己实现成了 **另一个 PrivateUse1 后端**，和 torch_fl 争同一把 key。
- 对 Ascend，现有路线（`csrc/aten/backends/ascend/` 手写/接 CANN 算子，或 FlagGems + triton-ascend）仍是应走的方向；`libtorch_npu.so` 无法作为"零成本兜底层"直接外挂。
- 若确实想复用 torch_npu 已实现的 NPU kernel，需要的不是"外挂 so"，而是**在 C++ 层显式转调 torch_npu 的 op 实现**（绕开 dispatcher 的 PrivateUse1 单键限制），这是另一套工程量，且强绑 torch_npu 版本，收益/代价需另行评估。

## 一句话总结

> `libtorch_npu.so` 能被 CPU torch 加载、符号也喂得饱，但它把 NPU 算子注册在 **PrivateUse1**——正是 torch_fl 的 `flagos` 已占用的 key。CUDA 方案依赖的"独立 key 分层"前提在 Ascend 上不存在，因此**"抽 so 外挂即兜底"在昇腾上不成立**。
