# 实测：CPU torch + 外挂 libtorch_cuda.so 复用 CUDA 算子

> 实测日期：2026-07-16
> 实测机器：2080ti（4× RTX 2080 Ti，driver 550.163.01）
> 结论：**成立**。pip 只装 CPU 版 torch、不装 CUDA torch，通过外挂一个版本匹配的 `libtorch_cuda.so`，即可复用 PyTorch 全套已注册的 CUDA kernel，真实计算结果正确。

## 背景与动机

torch_fl（`vm` / PrivateUse1 后端）希望：

- 在 NVIDIA 上**不手写 kernel**，直接复用 PyTorch 已优化好的 CUDA 算子（经 `device_boxing` 把 `vm` 张量改成 CUDA 元数据后调 `structured_*_out_cuda`）。
- 但**不想 pip 安装 CUDA 版 torch**（体积大、拉一堆 CUDA 依赖、且会把 Python 环境绑定到特定 CUDA 版本）。

核心疑问：**能否 pip 只装 CPU torch，另外单独挂一个 `libtorch_cuda.so` 进程内，让 CUDA kernel 注册进 dispatcher 供 boxing 使用？**

此前的推演倾向于"走不通"，理由是"CPU wheel（`USE_CUDA=0` 构建）的 `libtorch_cpu.so` 可能裁剪了 CUDA 相关符号，喂不饱 CUDA 构建产出的 `libtorch_cuda.so`"。**本次实测推翻了这个判断。**

## 实测环境搭建

```bash
# 1. 干净 conda 环境 + 只装 CPU torch
conda create -n libtorch_test python=3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu
#   → torch 2.13.0+cpu
#   torch/lib 下只有 libc10.so + libtorch_cpu.so，无任何 CUDA .so
#   torch.cuda.is_available() == False

# 2. 下载版本完全匹配的 CUDA wheel（只下载，不安装）
pip download torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126 -d /tmp/cuda_wheel --no-deps
#   解压 wheel，取出 libtorch_cuda.so (≈1GB) + libc10_cuda.so

# 3. 装 CUDA runtime 依赖库（这些是独立的 nvidia-* 包，不碰 torch 本体）
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
    nvidia-nvtx-cu12 nvidia-cuda-cupti-cu12 nvidia-cusparselt-cu12 \
    nvidia-nvjitlink-cu12 nvidia-cuda-cccl-cu12 nvidia-nvshmem-cu12
#   torch 仍是 2.13.0+cpu 不变
```

> **关键点：libtorch release 独立包（download.pytorch.org/libtorch/...）在 2.13.0 上 404，不可用；改用 `pip download` 拿 CUDA wheel 抽 .so 更可靠**——它和 CPU wheel 出自同一套 pip 构建体系，ABI 匹配度最高，且版本可逐位对齐。

## 四个关卡的实测结果

验证按依赖顺序分四关，逐一通过：

| # | 关卡 | 结果 | 说明 |
|---|---|---|---|
| 1 | **符号解析** | ✅ 无 undefined symbol | `ctypes.CDLL(libtorch_cuda.so, RTLD_GLOBAL)` 成功加载。CPU wheel 的 `libtorch_cpu.so`/`libc10.so` **完整满足** CUDA 构建 `libtorch_cuda.so` 的全部符号需求——**未发生符号裁剪** |
| 2 | **kernel 注册** | ✅ 加载后 CUDA key 全部就位 | 加载前 `mm/add/_softmax/bmm` 的 `CUDA=False`，加载后全部 `CUDA=True`。证明 dispatcher 是 `libc10.so` 的全局单例，谁加载 `.so`，kernel 就注册进那张表 |
| 3 | **CUDAHooks / 设备初始化** | ✅ 但**必须 `LD_PRELOAD`** | 见下方「关键约束」 |
| 4 | **真实计算** | ✅ 结果正确 | `mm max_err=9.5e-06`（fp32 GEMM 正常精度）、`add max_err=0.0`、`softmax rowsum=1.0`，确在 `cuda:0` 上执行 |

### 关卡 1/2 的验证（无副作用检查）

```python
import ctypes, torch
from torch._C import _dispatch_dump_table  # 简化示意
# 加载前：aten::mm 只有 CPU
ctypes.CDLL(".../libc10_cuda.so", ctypes.RTLD_GLOBAL)
ctypes.CDLL(".../libtorch_cuda.so", ctypes.RTLD_GLOBAL)
# 加载后：aten::mm / add / _softmax / bmm 均出现 CUDA 实现
```

前置：需先把所有 `nvidia/*/lib` 目录加进 `LD_LIBRARY_PATH`（cudart、cublas、cudnn、nvshmem 等），否则会因缺 `libcudart.so.12`、`libnvshmem_host.so.3` 等运行库而加载失败（**注意：这类失败是"缺 CUDA 运行库"，不是符号不匹配**）。

### 关卡 4 的验证（真实计算）

```python
# 关键：libtorch_cuda.so 需在 import torch 之前载入（见下方约束）
a = torch.empty([N, K], device='cuda')  # 走 factory 路径，C++ 层直接建，成功
# ... 填充数据、执行 mm/add/softmax ...
# mm  max_err = 9.5367431640625e-06
# add max_err = 0.0
# softmax rowsum mean = 1.0
```

## 关键约束

### 约束 1（硬约束）：libtorch_cuda.so 必须在 `import torch` 之前载入

- **现象**：若在 `import torch` **之后**再 `ctypes.CDLL(libtorch_cuda.so)`，虽然 kernel 注册进了 dispatcher（关卡 2 过），但设备初始化会报 `Cannot initialize CUDA without ATen_cuda library`。
- **根因**：PyTorch 的 **CUDAHooks 机制**——`getCUDAHooks()` 在 `import torch` 时首次调用并**缓存**了 `libtorch_cpu.so` 里的"桩 Hooks"（专门抛该错）；后加载的 `libtorch_cuda.so` 注册的真 Hooks 覆盖不掉已缓存的桩。
- **解法**：用 `LD_PRELOAD`（或构建期 rpath / 在 `import torch` 前 `ctypes.CDLL`）让 `libtorch_cuda.so` 先于 torch 载入。实测 `LD_PRELOAD=".../libc10_cuda.so:.../libtorch_cuda.so"` 后，连最苛刻的 factory 路径 `torch.empty(device='cuda')` 都成功返回 `cuda:0`。

### 约束 2（对 torch_fl 无影响）：Python 层 `torch.cuda._lazy_init` gate

- **现象**：`torch.randn(device='cuda')`、`a @ b`、`.to('cuda')`、`.copy_()` 等高层 API 会显式调用 `torch.cuda._lazy_init()`，撞上 `torch/cuda/__init__.py` 里的 `AssertionError: Torch not compiled with CUDA enabled`。这是 **Python 层的编译期旗标 gate**，与 C++ dispatcher 里有没有 CUDA kernel 无关。
- **对 torch_fl 无影响**：torch_fl 的 `vm`(PrivateUse1) + boxing 路径**从不调用 `torch.cuda.*`**——它用自己的 flagos allocator 分配显存，boxing 改元数据后直接在 C++ 层调 `structured_*_out_cuda`。因此这个 Python gate 天然被绕开。
- 纯 Python 复现时（本次实测）需短路该 gate 才能测到真实计算，这只是复现手段，不是 torch_fl 的真实约束。

### 约束 3：版本必须逐位匹配

`libtorch_cuda.so` 与 pip CPU torch 的版本必须**完全一致**（如 `2.13.0` 对 `2.13.0`，nightly 连日期都要对）。混版本会因 `at::Tensor`/ABI 布局差异导致符号或运行时错乱。

### 约束 4：依赖 CPU wheel 符号完整性（无官方承诺）

本方案依赖"CPU wheel 的 `libtorch_cpu.so` 符号足够喂饱 `libtorch_cuda.so`"这一性质。**PyTorch 未明文承诺此性质**——2.13.0 实测成立，但升级 torch 版本时应重跑关卡 1/2 复测一次。

## 对 `vm` 后端（torch_fl）的落地要点

1. **预载时机**：`torch_fl/__init__.py` 已在用 `ctypes.CDLL(..., RTLD_GLOBAL)` 预载 `libtorch.so`。把 `libc10_cuda.so` + `libtorch_cuda.so` 加入预载列表，并确保**在 `import torch` 之前**执行（或通过 `LD_PRELOAD` / 链接期 rpath 保证）。这是唯一的硬约束。
2. **boxing 路径可用**：`structured_*_out_cuda` 等已注册且能执行，`device_boxing.h`（flagos 自管显存 + 改元数据调 native kernel）成立。**#15 的 boxing/structured 复用成果无需回退。**
3. **CUDA runtime 依赖**：需要 `nvidia-*` pip 包提供 `libcudart/libcublas/libcudnn/libnvshmem` 等 `.so`，通过 `LD_LIBRARY_PATH` 或 rpath 定位。
4. **不碰 `torch.cuda` Python API**：保持 boxing 全程在 C++ 层，避免触发 `_lazy_init` gate。

## 换来了什么 / 代价

**换来：**
- ✅ 不 pip 装 CUDA torch，Python 侧保持干净的 `+cpu` 环境
- ✅ 复用 PyTorch 全套优化过的 CUDA kernel，`vm` 后端**零手写 kernel、无需写 cuBLAS/cuDNN 胶水**
- ✅ 跟得上 torch 最新版：换版本时外挂对应版本的 `libtorch_cuda.so` 即可

**代价：**
- ⚠️ 依赖"CPU wheel 符号完整"这一无官方承诺的性质（升级需复测，见约束 4）
- ⚠️ `LD_PRELOAD` / 预载时机是硬约束（约束 1）
- ⚠️ 进程内仍有 `libtorch_cuda.so`，仍 ABI-绑定该 torch 版本（这是"能跟上最新版"而非"一份二进制跨版本"）

## 一句话总结

> **pip 只装 CPU torch、外挂版本匹配的 `libtorch_cuda.so`（在 import torch 前载入），即可让 CUDA kernel 注册进 dispatcher 并被 `vm`/boxing 路径复用，实测计算正确。** 唯一硬约束是加载时机（CUDAHooks 缓存问题，用 `LD_PRELOAD` 解决）；Python 层的 `torch.cuda` gate 与 torch_fl 无关。这条路让 NVIDIA 后端零手写 kernel 且不依赖 pip CUDA torch。
