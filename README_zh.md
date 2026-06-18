# torch_fl

基于 PyTorch PrivateUse1 扩展机制的自定义设备插件，将 [FlagGems](https://github.com/FlagOpen/FlagGems) 高性能 Triton 算子注册为 `flagos` 设备后端，实现统一的多芯支持。

## 特性

- 自动将 FlagGems Triton 算子注册为 `flagos` 设备的 dispatch 实现
- 可配置的后端路由：按算子粒度选择 FlagGems 或 原始的厂商后端（CUDA/MetaX/Ascend）
- 目前支持 CUDA、MetaX 和 Ascend 三种硬件平台
- 完整的设备管理 API（stream、event、RNG、AMP）


## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.11.0 |
| CUDA | 12.8 |
| FlagGems | 5.0.2 |

> CUDA 12.2 存在已知的数值精度问题（NaN），请使用 12.9 或更高版本。

## 安装

### 前置依赖

- 硬件 Runtime 依赖：
    - CUDA toolkit 12.8 （仅在 CUDA 平台需要）
    - MetaX cu-bridge 库（仅在 MetaX 平台需要）
    - CANN toolkit（仅在 Ascend 平台需要）
- PyTorch 2.11.0
- FlagGems（5.0.2 版本以上，需要开启 DFLAGGEMS_BUILD_C_EXTENSIONS），源码安装参考文档：[FlagGems 安装](https://flagos-ai.github.io/FlagGems/getting-started/install/)
  - 注意：Ascend 平台上 FlagGems 为可选依赖

### 从源码安装（CUDA 平台）

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

ACCELERATOR=cuda FLAGGEMS_DIR=/path/to/FlagGems/build/cpython-312/ \
  FLAGGEMS_KERNEL=1 FLAGGEMS_PYTHON=1 CUDA_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

### 从源码安装（MetaX 平台）

MetaX 构建与 Ascend 类似：**主工程仅 CXX**，设备算子由 `mxcc`/`cucc` 编译 `csrc/aten/backends/metax/*.cu` 后以 object 链入 `libtorch_fl.so`；运行时走 `runtime/accelerator/metax`（cu-bridge），**不**委托 `at::cuda`/`at::maca`。

**前置依赖**

- MetaX MACA SDK（默认 `/opt/maca`），含 cu-bridge 与 `mxcc`/`cucc`
- 与 MetaX 栈匹配的 PyTorch wheel（见下方[运行时说明](#metax-运行时说明)）
- FlagGems 5.0.2+（可选；仅当算子路由到 `flagos_python` 时需要）

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

# MetaX SDK 路径（按实际安装位置调整）
export METAX_PATH=/opt/maca
export PATH=/opt/maca/tools/cu-bridge/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:$LD_LIBRARY_PATH

ACCELERATOR=metax METAX_KERNEL=ON FLAGGEMS_PYTHON=1 FLAGGEMS_KERNEL=0 CUDA_KERNEL=0 \
  pip install --no-build-isolation -vvv -e .
```

> 在 MetaX 上，PyPI 通用版 Triton（`nvidia` 后端）无法为 MetaX 硬件 JIT 内核。请使用 `torch_fl/backends_metax.conf` 或 `torch_fl/backends_metax_flagos_py.conf`，将不兼容算子路由到 metax C++ kernel（见[MetaX 后端配置](#metax-后端配置)）。

### 从源码安装（Ascend 平台）

```bash
# 确保 CANN toolkit 已安装并已 source 环境
# （通常: source /usr/local/Ascend/ascend-toolkit/set_env.sh）

ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 \
  CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

Ascend 平台上禁用 FlagGems C++ kernel 和 CUDA kernel，仅编译 Ascend kernel 后端（ACL NN API），同时启用 FlagGems Python 封装用于路由到 FlagGems 的算子。

### 构建环境变量

| 变量 | 说明 |
|------|------|
| `ACCELERATOR` | 硬件平台：`cuda`（默认）、`metax` 或 `ascend` |
| `FLAGOS_BUILD_JOBS` | 原生库并行编译线程数（默认 CPU 核数）；日志过长可设 `1` |
| `CUDA_HOME` | CUDA toolkit 路径 |
| `METAX_PATH` | MetaX SDK 路径（默认 `/opt/maca`，metax 构建必需） |
| `METAX_ARCH` / `METAX_MXCC` | 可选：GPU 架构或 mxcc/cucc 编译器路径 |
| `METAX_KERNEL` | 启用 MetaX C++ kernel 构建（`ON`/`OFF`；`ACCELERATOR=metax` 时自动开启） |
| `ASCEND_HOME` | CANN toolkit 路径（默认 `/usr/local/Ascend/ascend-toolkit/latest`） |
| `FLAGGEMS_DIR` | FlagGems C++ 库路径（启用低开销 C++ dispatch） |
| `FLAGGEMS_KERNEL` | 启用 FlagGems C++ kernel 封装（`ON`/`OFF`，默认 `ON`；Ascend 设为 `0`） |
| `FLAGGEMS_PYTHON` | 启用 FlagGems Python kernel 封装（`ON`/`OFF`，默认 `OFF`；设为 `1` 启用） |
| `CUDA_KERNEL` | 启用 CUDA kernel 构建（`ON`/`OFF`，默认 `ON`；Ascend 设为 `0`） |
| `ASCEND_KERNEL` | 启用 Ascend kernel 构建（`ON`/`OFF`，默认 `OFF`；Ascend 设为 `1`） |

### 运行时环境变量

| 变量 | 说明 |
|------|------|
| `FLAGOS_DISABLE_FLAGGEMS_PY` | 设为 `1` 关闭 FlagGems Python 层注册（C++ stub-only 模式） |
| `FLAGOS_METAX_CUDART_SHIM` | 设为 `1` 在 import 前加载 libcudart 兼容 shim（通用 PyTorch wheel 常需） |
| `FLAGOS_METAX_COMPAT` | 设为 `1` 为 FlagGems 修补 `torch.cuda` 设备属性查询 |
| `GEMS_VENDOR` | FlagGems 厂商名；MetaX 上设为 `metax` |
| `LD_PRELOAD` | 常设为 `/opt/maca/lib/libsymbol_cu.so`，用于 cu-bridge 符号解析 |
| `FLAGGEMS_SOURCE_DIR` | FlagGems 源码目录（算子路由到 `flaggems` 或 `flagos_python` 时需设置） |
| `FLAGOS_BACKEND_CONFIG` | 覆盖后端路由配置（MetaX：`backends_metax.conf` 或 `backends_metax_flagos_py.conf`） |
| `FLAGOS_LOG_DISPATCH` | 设为 `1` 打印每次算子 dispatch 的后端选择 |
| `FLAGOS_OP_<name>` | 按算子覆盖后端（算子名中的 `.` 替换为 `__`） |

## 使用

### 基本用法

```python
import torch
import torch_fl  # 导入即自动注册 FlagGems 算子

# 在 flagos 设备上创建 tensor
x = torch.randn(1000, 1000, device="flagos")
y = torch.randn(1000, 1000, device="flagos")

# 所有运算自动使用 FlagGems Triton 内核
z = x + y
mm_result = torch.mm(x, y)
softmax_result = torch.softmax(x, dim=-1)
```

### 设备间数据搬移

```python
cpu_tensor = torch.randn(3, 3)
flagos_tensor = cpu_tensor.to("flagos")
back_to_cpu = flagos_tensor.cpu()
```

### 设备上下文管理

```python
with torch_fl.flagos.device(0):
    a = torch.randn(10, 10, device="flagos")
```

### MetaX 平台导入顺序

在 MetaX 硬件上，**必须**在 `import torch` 之前导入 `torch_fl`：

```python
import torch_fl  # 必须先导入
import torch
```

原因：PyTorch 自带的 CUDA 12.x 运行时与 MetaX 的 cu-bridge（CUDA 11.6 兼容层）ABI 不兼容。`torch_fl` 会预加载一个 shim 库来提供所需的符号版本。

CUDA 平台无此限制。

### MetaX 运行时环境

运行测试或推理前，配置 SDK 路径与混合后端：

```bash
export METAX_PATH=/opt/maca
export PATH=/opt/maca/tools/cu-bridge/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/maca/lib/libsymbol_cu.so

export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_METAX_COMPAT=1
export GEMS_VENDOR=metax
export FLAGOS_BACKEND_CONFIG=torch_fl/backends_metax_flagos_py.conf
export FLAGGEMS_SOURCE_DIR=$(python -c "import os,flag_gems;print(os.path.dirname(flag_gems.__file__))")
```

#### MetaX 运行时说明

- **PyTorch + Triton 栈**：官方 `maca-pytorch` 镜像自带 `torch+metax` 与 `triton+metax`（输出 `mcfatbin`）。通用 PyTorch wheel + PyPI Triton 走 NVIDIA 后端，在 MetaX 上会报 `PTX JIT compilation failed`，需将相关算子路由到 metax C++ kernel。
- **`flash_attn`**：预编译 MetaX `flash_attn` wheel 可能与较新 PyTorch ABI 不兼容；加载 Qwen3/transformers 前需禁用或 patch。
- **`relu` / `sigmoid`**：当前树中未通过 `m.impl` 注册，走 cpu_fallback；除非已在 `MetaxKernels.cmake` 中启用 GPU kernel，否则不要在配置里写 `metax`。

### C++ Stub-Only 模式

可以完全关闭 FlagGems Python 层注册，仅使用 C++ 统一 wrapper 进行算子 dispatch。适用于验证 C++ stub 覆盖度是否完整。

```bash
# 必须：告知 FlagGems C++ native API Triton kernel 源码位置
export FLAGGEMS_SOURCE_DIR=$(python -c "import os;import flag_gems;print(os.path.dirname(flag_gems.__file__))")

python your_script.py
```

此模式下所有算子 dispatch 由 C++ dispatch stub（`backends.conf` 路由）处理，不经过 FlagGems 的 Python `torch.library` 注册。

### 查询状态

```python
torch_fl.flagos.is_available()       # 设备是否可用
torch_fl.flagos.device_count()       # 设备数量
torch_fl.flagos.current_device()     # 当前设备索引
torch_fl.flagos.synchronize()        # 同步设备
torch_fl.is_flaggems_enabled()       # FlagGems 算子是否已注册
torch_fl.get_registered_ops()        # 已注册的算子列表
```

## 后端配置

可以按算子粒度配置使用 FlagGems 还是 CUDA 后端执行。

### 配置文件

默认路径 `torch_fl/backends.conf`，可通过 `FLAGOS_BACKEND_CONFIG` 环境变量覆盖：

```ini
# 格式: op_name = backend
# backend: "flagos" | "flaggems" | "cuda"
# 未列出的算子默认使用 flagos (FlagGems)
mm = cuda
bmm = flagos
cat = cuda
```

### 环境变量覆盖

单个算子可通过环境变量覆盖配置文件（优先级更高）：

```bash
# 格式: FLAGOS_OP_<op_name>=cuda|flaggems
# 算子名中的 "." 替换为 "__"
export FLAGOS_OP_mm=cuda
export FLAGOS_OP_mm__out=cuda
```

### MetaX 后端配置

| 文件 | 用途 |
|------|------|
| `torch_fl/backends_metax.conf` | 所列算子全部 → `metax` C++ kernel。pytest 检测到 MetaX（`/dev/mxcd`）且未设置 `FLAGOS_BACKEND_CONFIG` 时自动选用。 |
| `torch_fl/backends_metax_flagos_py.conf` | **集成测试推荐。** 混合路由：多数计算算子 → `flagos_python`；将 Triton 不兼容算子（`mm`/`bmm`/`mean.dim`）以及分配/工厂算子（`zeros`、`scalar_tensor`、`embedding` 等）保留在 `metax`。 |

示例（`backends_metax_flagos_py.conf`）：

     # elementwise / inference-path ops
     abs = flagos_python
     add.Tensor = flagos_python
     cos = flagos_python
     sin = flagos_python     
     
     # Triton 不兼容
     mm = metax
     bmm = metax
     mean.dim = metax
     # 分配/工厂算子
     zeros = metax
     scalar_tensor = metax

### 调试 dispatch

```bash
export FLAGOS_LOG_DISPATCH=1  # 打印每次算子 dispatch 的后端选择
```

## 测试

`tests/integration/ops/` 下的测试通过 `@pytest.mark` 标记平台分类：

| 标记 | 含义 | 运行时机 |
|------|------|----------|
| `@pytest.mark.anyplatform` | 正确性测试，所有平台都应运行 | 始终 |
| `@pytest.mark.cuda` | CUDA/FlagGems dispatch 路由测试 | 仅 CUDA 平台 |
| `@pytest.mark.ascend` | Ascend 后端 dispatch 测试 | 仅 Ascend 平台 |

### CUDA 平台

```bash
# 算子测试（需要 FlagGems 源码用于 C++ native API）
FLAGGEMS_SOURCE_DIR=/path_to_repos/FlagGems/src/flag_gems \
  pytest tests/integration/ops/ -v -m "anyplatform or cuda"

# Qwen3 推理测试
FLAGGEMS_SOURCE_DIR=/path_to_repos/FlagGems/src/flag_gems \
  pytest tests/integration/test_qwen3_infer.py -v -s

# Qwen3 训练测试（单卡）
FLAGGEMS_SOURCE_DIR=/path_to_repos/FlagGems/src/flag_gems \
  pytest tests/integration/test_qwen3_train.py -v -s --steps 10

# 仅运行 CUDA 相关测试
pytest tests/integration/ops/ -v -m cuda

# 仅运行 FlagGems (Triton) 后端测试
pytest tests/integration/ops/ -v -m flaggems

# 仅运行 FlagGems Python wrapper 测试
pytest tests/integration/ops/ -v -m flaggems_python

# 仅运行平台无关的正确性测试
pytest tests/integration/ops/ -v -m anyplatform

# FlagGems Python wrapper (flagos_python) 端到端测试
FLAGOS_BACKEND_CONFIG=torch_fl/backends_flagos_py.conf \
  pytest tests/integration/ops/ -v
```

### MetaX 平台

```bash
# 运行时环境（见上文「MetaX 运行时环境」）
export METAX_PATH=/opt/maca
export PATH=/opt/maca/tools/cu-bridge/bin:/opt/maca/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:/opt/maca/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/maca/lib/libsymbol_cu.so
export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_METAX_COMPAT=1
export GEMS_VENDOR=metax
export FLAGOS_BACKEND_CONFIG=torch_fl/backends_metax_flagos_py.conf
export FLAGGEMS_SOURCE_DIR=$(python -c "import os,flag_gems;print(os.path.dirname(flag_gems.__file__))")

# 基础算子测试（含 Qwen3 推理路径：cos/sin/rsqrt/silu 等）
pytest tests/integration/test_ops.py -v

# 逐算子 dispatch 测试（混合配置）
pytest tests/integration/ops/ -v

# Qwen3 推理
pytest tests/integration/test_qwen3_infer.py -v -s --model /path/to/Qwen3-0.6B

# Qwen3 训练（单卡）
pytest tests/integration/test_qwen3_train.py -v -s --steps 10

# 纯 metax C++ kernel 模式（不走 flagos_python）
FLAGOS_BACKEND_CONFIG=torch_fl/backends_metax.conf \
  FLAGOS_DISABLE_FLAGGEMS_PY=1 \
  pytest tests/integration/test_ops.py -v
```

未设置 `FLAGOS_BACKEND_CONFIG` 时，`tests/integration/conftest.py` 会在 MetaX 硬件上自动选择 `torch_fl/backends_metax.conf`。

### Ascend 平台

```bash
# 算子测试
FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf \
  pytest tests/integration/ops/ -v -m "anyplatform or ascend"

# Qwen3 推理测试
FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf \
  pytest tests/integration/test_qwen3_infer.py -v -s

# Qwen3 训练测试（单卡）
FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf \
  pytest tests/integration/test_qwen3_train.py -v -s --steps 10
```

`test_qwen3_infer.py` 和 `test_qwen3_train.py` 在所有平台上使用相同代码，仅安装方式（`ACCELERATOR=ascend pip install -e .`）和运行时环境变量不同。

### Pytest Marks

`tests/integration/ops/` 中的算子测试使用 pytest mark 标记平台/后端依赖：

| Mark | 说明 |
|------|------|
| `@pytest.mark.anyplatform` | 平台无关的正确性测试（shape、dtype、broadcast） |
| `@pytest.mark.cuda` | 需要 CUDA 后端或 CUDA 参考对比 |
| `@pytest.mark.flaggems` | 需要 FlagGems (Triton) 后端 |
| `@pytest.mark.flaggems_python` | 需要 FlagGems Python wrapper (pybind11 路径) |
| `@pytest.mark.ascend` | 需要 Ascend NPU 后端 |

使用 `-m <mark>` 运行特定类别的测试，例如：`pytest tests/integration/ops/ -m cuda` 仅运行 CUDA 测试。

## 项目结构

```
PyTorch-Plugin-FL/
├── include/                  # 公共头文件
│   ├── flagos.h              #   统一 runtime API（memory、stream、device）
│   └── macros.h              #   通用宏定义
├── csrc/
│   ├── aten/                 # ATen 算子层
│   │   ├── common.{h,cc}     #   后端配置加载、Backend 枚举
│   │   ├── dispatcher.h      #   轻量算子 dispatcher（替代 PyTorch DispatchStub）
│   │   ├── device_boxing.h   #   零拷贝 flagos↔CUDA tensor 元数据转换
│   │   ├── register.cc       #   PrivateUse1 dispatch key 注册
│   │   ├── {op}.{h,cc}       #   各算子 stub 定义（add、mm、silu 等）
│   │   └── backends/         #   后端特定 kernel 实现
│   │       ├── cuda/         #     CUDA kernel（cuBLAS、修改版 PyTorch kernel）
│   │       ├── flagos/       #     FlagGems C++ native API wrapper
│   │       └── ascend/       #     Ascend kernel（ACL NN API）
│   └── runtime/              # 设备运行时
│       ├── device_allocator  #   设备内存分配器
│       ├── host_allocator    #   pinned memory 分配器
│       ├── guard             #   DeviceGuard 实现
│       ├── generator         #   RNG generator
│       ├── hooks             #   运行时 hook
│       └── accelerator/      #   硬件抽象层
│           ├── cuda/         #     CUDA runtime 实现
│           ├── maca/         #     MACA cudart shim（符号版本兼容）
│           └── ascend/       #     Ascend runtime（基于 ACL 的 memory、stream、device）
├── torch_fl/
│   ├── __init__.py           # 插件入口：注册设备、加载 FlagGems 算子
│   ├── flagos/               # Python 设备模块（stream、event、RNG、AMP）
│   ├── accelerator/          # Python accelerator 模块（MACA shim 加载器）
│   ├── backends.conf                  # 默认后端路由配置（CUDA/FlagGems）
│   ├── backends_metax.conf            # MetaX：所列算子 → metax
│   ├── backends_metax_flagos_py.conf  # MetaX 混合：metax + flagos_python
│   ├── backends_flagos_py.conf        # FlagGems Python 封装路由
│   ├── backends_ascend.conf           # Ascend 后端路由（所有算子 → ascend）
│   ├── distributed.py        # 分布式训练支持（DDP patch）
│   ├── integration.py        # FlagGems 算子注册逻辑
│   ├── csrc/                 # C 扩展（module.cc、stub.c）
│   └── lib/                  # 编译后的共享库（libtorch_fl.so、libflagos.so）
├── tests/
│   ├── integration/          # 自动化集成测试
│   │   ├── ops/              #   各算子 dispatch 测试
│   │   ├── test_qwen3_*.py   #   端到端模型测试
│   │   └── conftest.py       #   Pytest 配置
│   ├── manual/               # 手动测试脚本
│   └── common/               # 测试工具
├── debug/                    # 开发笔记和调试脚本
├── cmake/                    # CMake 模块
├── setup.py                  # CMake 构建入口
└── pyproject.toml
```

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│  Python: import torch_fl                                     │
│  ┌────────────────┐  ┌────────────────────────────┐          │
│  │ torch_fl.flagos│  │ torch_fl.distributed       │          │
│  │ (device API)   │  │ (DDP/FSDP patch)           │          │
│  └────────────────┘  └────────────────────────────┘          │
├──────────────────────────────────────────────────────────────┤
│  PrivateUse1 Dispatch                                        │
│  ┌─────────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐    │
│  │ FlagGems    │  │ CUDA     │  │ Ascend    │  │ CPU    │    │
│  │ (Triton)    │  │ (native) │  │ (ACL NN)  │  │fallback│    │
│  └─────────────┘  └──────────┘  └───────────┘  └────────┘    │
├──────────────────────────────────────────────────────────────┤
│  C++ Runtime (csrc/)                                         │
│  ┌──────────┐ ┌────────┐ ┌───────┐ ┌───────────┐             │
│  │Allocator │ │ Guard  │ │ RNG   │ │ Hooks     │             │
│  └──────────┘ └────────┘ └───────┘ └───────────┘             │
├──────────────────────────────────────────────────────────────┤
│  Hardware Abstraction (accelerator/)                         │
│  ┌──────────────┐  ┌─────────────────────┐  ┌────────────┐   │
│  │ CUDA Runtime │  │ MetaX cu-bridge+shim │  │ Ascend ACL │   │
│  └──────────────┘  └─────────────────────┘  └────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 许可证

Apache-2.0
