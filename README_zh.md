# torch_fl

基于 PyTorch PrivateUse1 扩展机制的自定义设备插件，将 [FlagGems](https://github.com/FlagOpen/FlagGems) 高性能 Triton 算子注册为 `flagos` 设备后端，实现统一的多芯支持。

## 特性

- 自动将 FlagGems Triton 算子注册为 `flagos` 设备的 dispatch 实现
- 可配置的后端路由：按算子粒度选择 FlagGems 或 原始的厂商后端（CUDA/MACA）
- 目前支持 CUDA 和 MACA (MetaX) 两种硬件平台
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
    - MACA cu-bridge 库（仅在 MACA 平台需要）
- PyTorch 2.11.0
- FlagGems（5.0.2 版本以上，需要开启 DFLAGGEMS_BUILD_C_EXTENSIONS），源码安装参考文档：[FlagGems 安装](https://flagos-ai.github.io/FlagGems/getting-started/install/)

### 从源码安装（CUDA 平台）

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

pip install -e . --no-build-isolation
```

### 从源码安装（MACA 平台）

```bash
# 设置 MACA cu-bridge 库路径，取决于实际环境中的cu-bridge路径
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:$LD_LIBRARY_PATH

ACCELERATOR=maca pip install -e . --no-build-isolation
```

### 构建环境变量

| 变量 | 说明 |
|------|------|
| `ACCELERATOR` | 硬件平台，`cuda`（默认）或 `maca` |
| `CUDA_HOME` | CUDA toolkit 路径 |
| `MACA_PATH` | MACA SDK 路径（默认 `/opt/maca`） |
| `FLAGGEMS_DIR` | FlagGems C++ 库路径（启用低开销 C++ dispatch） |

### 运行时环境变量

| 变量 | 说明 |
|------|------|
| `FLAGOS_DISABLE_FLAGGEMS_PY` | 设为 `1` 关闭 FlagGems Python 层注册（C++ stub-only 模式） |
| `FLAGGEMS_SOURCE_DIR` | FlagGems 源码目录（当 C++ native API 算子路由到 `flaggems` 后端时必须设置） |
| `FLAGOS_BACKEND_CONFIG` | 覆盖 `backends.conf` 路径 |
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

### MACA 平台导入顺序

在 MetaX (MACA) 硬件上，**必须**在 `import torch` 之前导入 `torch_fl`：

```python
import torch_fl  # 必须先导入
import torch
```

原因：PyTorch 自带的 CUDA 12.x 运行时与 MACA 的 cu-bridge（CUDA 11.6 兼容层）ABI 不兼容。`torch_fl` 会预加载一个 shim 库来提供所需的符号版本。

CUDA 平台无此限制。

### C++ Stub-Only 模式

可以完全关闭 FlagGems Python 层注册，仅使用 C++ 统一 wrapper 进行算子 dispatch。适用于验证 C++ stub 覆盖度是否完整。

```bash
# 必须：告知 FlagGems C++ native API Triton kernel 源码位置
export FLAGGEMS_SOURCE_DIR=$(python -c "import os;import flag_gems;print(os.path.dirname(flag_gems.__file__))")

# 关闭 FlagGems Python 层注册
export FLAGOS_DISABLE_FLAGGEMS_PY=1

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

### 调试 dispatch

```bash
export FLAGOS_LOG_DISPATCH=1  # 打印每次算子 dispatch 的后端选择
```

## 测试

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 基础算子测试
pytest tests/integration/test_ops.py -v --device cuda
pytest tests/integration/test_ops.py -v --device flagos

# dispatch 路由测试
pytest tests/integration/ops/ -v

# CPU fallback 追踪测试
pytest tests/integration/test_fallback_trace.py -v

# Qwen3 推理
pytest tests/integration/test_qwen3_infer.py -v -s --device cuda
pytest tests/integration/test_qwen3_infer.py -v -s --device flagos

# Qwen3 训练（单卡）
pytest tests/integration/test_qwen3_train.py -v -s --device cuda --steps 10
pytest tests/integration/test_qwen3_train.py -v -s --device flagos --steps 10
```

## 项目结构

```
PyTorch-Plugin-FL/
├── accelerator/              # 硬件抽象层
│   ├── include/flagos.h      #   统一 runtime API（memory、stream、device）
│   ├── csrc/cuda/            #   CUDA runtime 实现
│   └── csrc/maca/            #   MACA cudart shim（符号版本兼容）
├── csrc/
│   ├── aten/                 # ATen 算子层
│   │   ├── common.{h,cc}    #   后端配置加载、FlagosDevice 枚举
│   │   ├── dispatch_stub.h   #   轻量 dispatch stub（替代 PyTorch DispatchStub）
│   │   ├── device_boxing.h   #   零拷贝 flagos↔CUDA tensor 元数据转换
│   │   ├── register.cc       #   PrivateUse1 dispatch key 注册
│   │   ├── factory_ops/      #   基础算子（empty、copy、contiguous、set、fallback）
│   │   ├── functional_ops/   #   计算算子（mm、bmm、cat、embedding、softmax 等）
│   │   └── native/cuda/      #   修改版 CUDA kernel（Loops.cuh 放宽设备检查）
│   └── runtime/              # 设备运行时
│       ├── device_allocator  #   设备内存分配器
│       ├── host_allocator    #   pinned memory 分配器
│       ├── guard             #   DeviceGuard 实现
│       └── generator         #   RNG generator
├── torch_fl/
│   ├── __init__.py           # 插件入口：注册设备、加载 FlagGems 算子
│   ├── flagos/               # Python 设备模块（stream、event、RNG、AMP）
│   ├── distributed.py        # 分布式训练支持（DDP patch）
│   ├── integration.py        # FlagGems 算子注册逻辑
│   └── csrc/                 # C 扩展（module.cc、stub.c）
├── tests/
│   ├── integration/          # 自动化集成测试
│   └── manual/               # 手动测试脚本
├── setup.py                  # CMake 构建入口
└── pyproject.toml
```

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│  Python: import torch_fl                             │
│  ┌────────────────┐  ┌────────────────────────────┐  │
│  │ torch_fl.flagos│  │ torch_fl.distributed       │  │
│  │ (device API)   │  │ (DDP/FSDP patch)           │  │
│  └────────────────┘  └────────────────────────────┘  │
├──────────────────────────────────────────────────────┤
│  PrivateUse1 Dispatch                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐    │
│  │ FlagGems    │  │ CUDA backend │  │ CPU       │    │
│  │ (Triton)    │  │ (native)     │  │ fallback  │    │
│  └─────────────┘  └──────────────┘  └───────────┘    │
├──────────────────────────────────────────────────────┤
│  C++ Runtime (csrc/)                                 │
│  ┌──────────┐ ┌────────┐ ┌───────┐ ┌───────────┐     │
│  │Allocator │ │ Guard  │ │ RNG   │ │ Hooks     │     │
│  └──────────┘ └────────┘ └───────┘ └───────────┘     │
├──────────────────────────────────────────────────────┤
│  Hardware Abstraction (accelerator/)                 │
│  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ CUDA Runtime     │  │ MACA cu-bridge + shim   │   │
│  └──────────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## 许可证

Apache-2.0
