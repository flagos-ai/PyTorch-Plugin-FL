<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

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
- FlagGems（5.0.2 版本以上）
  - CUDA 平台：从 [FlagGems 官方仓库](https://github.com/FlagOpen/FlagGems) 安装，需开启 `FLAGGEMS_BUILD_C_EXTENSIONS`
  - Ascend 平台：从 [Hchnr/FlagGems](https://github.com/Hchnr/FlagGems) 的 `torch_fl` 分支安装，需指定 `FLAGGEMS_BACKEND=FLAGOS`（详见下方 Ascend 安装步骤）

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

> 在 MetaX 上，PyPI 通用版 Triton（`nvidia` 后端）无法为 MetaX 硬件 JIT 内核。请使用 `torch_fl/configs/backends_metax.conf` 或 `torch_fl/configs/backends_metax_flagos_py.conf`，将不兼容算子路由到 metax C++ kernel（见[MetaX 后端配置](#metax-后端配置)）。

### 从源码安装（Ascend 平台）

#### 1. 安装 FlagGems（FLAGOS 后端）

Ascend 平台上 FlagGems 需要使用我们 fork 的 `torch_fl` 分支，并以 `FLAGGEMS_BACKEND=FLAGOS` 编译。这样 FlagGems 不依赖 `torch_npu` / `libtorch_npu.so`，而是通过 `torch_fl` 提供的 `GetCurrentStream` C API 获取 ACL stream。

```bash
# 克隆 FlagGems（torch_fl 分支）
git clone -b torch_fl https://github.com/Hchnr/FlagGems.git
cd FlagGems

# 确保 CANN toolkit 环境已激活
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 FlagGems（指定 FLAGOS 后端，跳过 C++ 扩展编译）
pip install --no-build-isolation -e . \
  --config-settings=cmake.define.FLAGGEMS_BACKEND=FLAGOS \
  --config-settings=cmake.define.FLAGGEMS_BUILD_C_EXTENSIONS=OFF

cd ..
```

> **为什么用 FLAGOS 后端？**
> FlagGems 的 `ascend/npu` 后端会链接 `libtorch_npu.so`，而我们的环境没有 `torch_npu`（`torch_fl` 本身就是 PrivateUse1 后端）。
> `FLAGOS` 后端通过 `extern "C" void* GetCurrentStream(int)` 获取 stream，由 `torch_fl` 的 `libstream_api.so` 提供实现，完全绕开 `torch_npu` 依赖。

#### 2. 安装 torch_fl

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

# 确保 CANN toolkit 环境已激活
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 \
  CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

说明：
- `FLAGGEMS_KERNEL=0`：禁用 FlagGems C++ kernel 封装（FLAGOS 后端暂不编译 `liboperators.so`）
- `FLAGGEMS_PYTHON=1`：启用 FlagGems Python 封装，通过 `python_wrapper` 机制调用 FlagGems Triton kernel
- `ASCEND_KERNEL=1`：编译 Ascend C++ 算子后端（ACL NN API）

#### 3. Patch triton-ascend

原版 triton-ascend 依赖 `torch_npu` / `libtorch_npu.so`。由于 `torch_fl` 替代了 `torch_npu` 作为 PrivateUse1 后端，需要 patch triton-ascend 使其使用 `flagos` 设备接口：

```bash
python scripts/patch_triton_ascend.py
```

脚本会自动检测 triton 安装路径并应用修改。脚本幂等，可重复执行。patch 后请清理 kernel 缓存：

```bash
rm -rf ~/.triton/cache/
```

#### 4. 验证安装

```bash
python -c "
import torch_fl
print('device count:', torch_fl.flagos.device_count())
print('FlagGems enabled:', torch_fl.is_flaggems_enabled())
print('registered ops:', len(torch_fl.get_registered_ops()))
"
```

#### 5. 运行推理测试

```bash
pytest tests/integration/test_qwen3_infer.py -v -s --model /path/to/Qwen3-0.6B
```

#### 6. 运行训练测试

```bash
pytest tests/integration/test_qwen3_train.py -v -s --model /path/to/Qwen3-0.6B
```

> **常见问题：`libtorch_npu.so: cannot open shared object file`**
>
> 如果遇到此错误，说明 triton-ascend 仍在尝试加载 `torch_npu`。请确认：
> 1. 安装 triton-ascend 后执行了 `python scripts/patch_triton_ascend.py`
> 2. FlagGems 是从 `https://github.com/Hchnr/FlagGems` 的 `torch_fl` 分支安装的
> 3. 安装时指定了 `FLAGGEMS_BACKEND=FLAGOS`
> 4. 已清理 triton kernel 缓存（`rm -rf ~/.triton/cache/`）

### 从源码安装（PPU 平台）

PPU（`PPU_SDK`）以 **CUDA 兼容** 形态呈现，因此直接复用 CUDA 构建，无需单独的
`ACCELERATOR=ppu` 分支。PPU 的 `torch` wheel 本身就是完整的 CUDA 版本
（`torch.version.cuda == '13.0'`、`torch.cuda.is_available() == True`），而
`PPU_SDK/CUDA_SDK` 是完整的 CUDA 13 toolkit（nvcc、头文件、`libcudart.so.13`）。
这使 PPU 成为 boxing 路线里最省事的情况：

- **无需 stock `+cpu` wheel、也无需外挂 `libtorch_cuda.so`**——PPU torch 自带 CUDA
  运行时，随 `import torch` 正常加载。
- PPU 的算子注册在独立的 `CUDA` dispatch key（而非 `PrivateUse1`），因此生成的 CUDA
  boxing 内核（`csrc/aten/generated/cuda_kernels.cc`，PrivateUse1 → CUDA）零改动复用。

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

# 纯 CUDA-boxing 构建（不接 FlagGems）。CUDA_HOME 指向 PPU 的 CUDA_SDK；
# FLAGOS_SKIP_CUDA_ASSETS=1 既跳过外挂 libtorch_cuda.so 的打包，也跳过固定的
# nvidia-*-cu12 运行时依赖（PPU 由 PPU_SDK/CUDA_SDK 提供 CUDA 13）。
ACCELERATOR=cuda \
  CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  CUDA_KERNEL=ON FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF \
  FLAGOS_SKIP_CUDA_ASSETS=1 \
  pip install --no-build-isolation -vvv -e .
```

运行时设置 `FLAGOS_DISABLE_CUDA_ASSETS=1`，使 import 期对打包版 `libtorch_cuda.so`
的预载成为 no-op（本就没有打包——由 PPU torch 自身提供）：

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 python -c "import torch_fl, torch; \
  x = torch.randn(4, 4, device='flagos'); \
  print((x @ x).cpu())"
```

**可选：PPU 上启用 FlagGems。** 构建时设 `FLAGGEMS_PYTHON=ON`（默认即为 ON），运行时设
`FLAGOS_USE_FLAGGEMS=1`；`import torch_fl` 会自动选择 `backends_flaggems.conf`，把已发现
的算子路由到 FlagGems 的 Triton 内核。PPU 不需要额外的兼容层，通用 CUDA shim 即可：
`libcuda.so` 是真实驱动，`is_nvidia_cuda_available()` 成立，于是设置
`GEMS_VENDOR=nvidia`，`triton.language.extra.cuda.libdevice` 也能正常解析（其中有 `pow`）。

PPU 的 Triton 来自厂商私有 index 而非 PyPI，其版本号（`3.5.0+v0.2.0.ppu2.1.0`）不满足
`triton>=3.5.1` 的约束。因此当检测到 `PPU_SDK` 时，`setup.py` 会移除 `flag_gems`/`triton`
依赖，改由用户自行安装：

```bash
pip install triton==3.5.0+v0.2.0.ppu2.1.0   # 厂商 index，见下方说明
pip install flag_gems

FLAGOS_DISABLE_CUDA_ASSETS=1 FLAGOS_USE_FLAGGEMS=1 python -c "import torch_fl, torch; \
  x = torch.randn(256, 256, device='flagos'); \
  print(torch.allclose(torch.softmax(x, -1).cpu(), torch.softmax(x.cpu(), -1), atol=1e-3))"
```

> **排查：安装 PPU Triton 时报 `Invalid cross-device link`**
>
> 厂商的 `triton` sdist 实际是个下载器，它先取回真实 wheel，再 `rename()` 到 pip 缓存目录。
> 若 pip 缓存与构建目录不在同一文件系统（例如缓存在 NFS、构建在 `/tmp`），该 rename 会以
> `[Errno 18]` 失败。此时用 `curl` 手动下载它打印的 wheel 地址
> （`Guessing wheel URL: ...`），再对该文件执行 `pip install` 即可。

**可选：PPU 上启用 FlagCX。** 分布式训练默认即可用，走 NCCL 兜底：`PPU_SDK` 自带
厂商适配的 `libnccl.so.2`，PPU 的 torch wheel 是 `USE_NCCL=1` 构建，因此
`ProcessGroupNCCL` 原生存在，`ProcessGroupFlagOS` 直接在零拷贝 CUDA view 上使用它。
若要改用 FlagCX（异构统一通信库），需从源码构建：

```bash
git clone https://github.com/FlagOpen/FlagCX.git && cd FlagCX
# --depth 1 不会拉取 submodule；缺少 third-party/json 会导致构建失败：
# "nlohmann/json.hpp: No such file or directory"
git submodule update --init --depth 1 third-party/json

make -j16 USE_PPU=1 \
  DEVICE_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  CCL_HOME=/usr/local/PPU_SDK/CUDA_SDK

# torch plugin 用 *nvidia* adaptor 构建，而非自动探测出的 `ppu`。两者生成的 torch
# 侧代码完全一致（PPU 是 CUDA-ABI：同样的 CUDAStreamGuard、CUDAEvent、
# devName="cuda"），但 FlagCX 只为 NVIDIA 和 MetaX adaptor 编译 extended_api
# creator，所以 `nvidia` 能拿到更完整的 ProcessGroupFlagCX 绑定。而 `ppu` 目前
# 还编不过 —— PPU 未被加入 plugin 的各 adaptor #ifdef 分支。
cd plugin/torch && FLAGCX_ADAPTOR=nvidia FLAGCX_HOME=$(git rev-parse --show-toplevel) \
  python setup.py install
```

之后 `import torch_fl` 会自动优先使用 FlagCX（`_try_build_flagcx` 在 NCCL 兜底之前
执行）；除了把 `libflagcx.so` 放到加载路径上，无需额外环境变量：

```bash
LD_LIBRARY_PATH=/path/to/FlagCX/build/lib:$LD_LIBRARY_PATH \
  python tests/manual/test_flagos_dist_live.py --world-size 4
```

**运行测试：**

```bash
# 纯 boxing
FLAGOS_DISABLE_CUDA_ASSETS=1 pytest tests/unit tests/integration/ops \
  tests/integration/test_factory_ops.py -q -m "not flaggems and not flaggems_python"

# FlagGems 路线（首次很慢：Triton 需要逐个编译并 autotune）
FLAGOS_DISABLE_CUDA_ASSETS=1 FLAGOS_USE_FLAGGEMS=1 pytest tests/integration/ops -q

# 分布式（collectives + DDP）。默认走 NCCL；用 FlagCX 时加上 LD_LIBRARY_PATH。
python tests/manual/test_flagos_dist_live.py --world-size 4
```

### 从源码安装（海光 DCU 平台）

海光 DCU（DTK）复用 **CUDA boxing 路线**，通过独立的 `ACCELERATOR=dcu` 分支支持。
之所以可行，取决于厂商栈的两个特性：

- DCU 的 `torch` wheel 是 **hipify 构建**：HIP 算子注册在 `CUDA` dispatch key 上，
  张量的设备类型也报告为 `DeviceType::CUDA`（`torch.version.cuda is None`，
  `torch.version.hip == '6.3.x'`）。因此生成的 PrivateUse1 → CUDA boxing 算子
  （`csrc/aten/generated/cuda_kernels.cc`）无需改动即可分发进 `libtorch_hip.so`。
- DTK 在 `$DTK_ROOT/cuda/cuda-*` 提供 **CUDA 兼容 toolkit**，其 `libcudart.so.12`
  只是 `libgalaxyhip.so` 之上的一层薄壳——与 `libtorch_hip.so` 使用的是同一个
  runtime，因此只有一份驱动状态，不会出现两套。`csrc/runtime/accelerator/cuda/`
  下的 runtime 源码可以用普通 host `g++` 原样编译：不需要 `nvcc`、`hipcc`，也不
  需要 hipify。

该构建是 **纯 boxing**：`CUDA_KERNEL`、`FLAGGEMS_KERNEL`、`FLAGGEMS_PYTHON` 全部
强制关闭（DTK 自带 Triton，PyPI 上面向 NVIDIA 的 `triton` wheel 是错误产物，因此
不会被拉取）。

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

source /opt/dtk/env.sh          # 会导出 ROCM_PATH；DTK_ROOT 同样生效

ACCELERATOR=dcu pip install --no-build-isolation -vvv -e .
```

`DTK_ROOT` 的解析顺序为 `DTK_ROOT` → `ROCM_PATH` → `/opt/dtk`。若 DTK 装在其他
位置，显式传入即可。

**验证：**

```bash
python -c "
import torch, torch_fl
print('device count:', torch.flagos.device_count())
x = torch.randn(512, 512, device='flagos')
print('mm matches .cuda():',
      torch.allclose(torch.mm(x, x).cpu(), torch.mm(x.cpu().cuda(), x.cpu().cuda()).cpu()))
"
```

**跑测试**（纯 boxing 构建，需要反选 FlagGems 相关 marker）：

```bash
pytest tests/unit tests/integration/test_allocator.py tests/integration/test_factory_ops.py -q
pytest tests/integration/ops -q -m "not flaggems and not flaggems_python"
```

说明：

- **内存池。** flagos 张量与 boxing 算子的输出共用一个池。`dcu_memory.h` 通过设备
  无关的注册表（`c10::getDeviceAllocator(kCUDA)`）把 caching 委托给 torch 自己的
  分配器，而不是走 `c10::cuda::` 命名空间——DCU wheel 导出的是
  `c10::hip::HIPCachingAllocator`，完全没有 `c10::cuda` 符号，而且
  `cuda_runtime.h` 与 `hip/hip_runtime.h` 无法出现在同一个编译单元里。因此
  `memory_allocated()` / `memory_reserved()` / `empty_cache()` 统计与行为都是真实的。
- **`record_stream` 在 DCU 上是 no-op**：从裸 stream 句柄构造 `c10::Stream` 需要
  `c10::cuda::getStreamFromExternal`，该 wheel 未导出此符号。
- **`.cuda()` 的反向与 `torch_fl` 不能在同一进程中共存。** PyTorch 的
  `register_privateuse1_backend` 会让 `at::getAccelerator()` 返回 `PrivateUse1`，
  于是 autograd 引擎在纯 CUDA 图上找不到 stream 元数据，在 `engine.cpp` 里断言失败。
  这是上游 PrivateUse1 的既有行为，在 CUDA/MetaX/Ascend 上表现一致——flagos 设备
  自身的反向不受影响。取 `.cuda()` 基准请放到独立进程里。
- **DTK 导出的 MIOpen CMake 配置** 内嵌了 `/usr/lib/x86_64-linux-gnu/librt.so`
  这个绝对路径，而 glibc ≥ 2.34 已把 librt 合并进 libc，该文件不再存在。
  `ACCELERATOR=dcu` 分支会把这类悬空绝对路径改写为 `-lrt`。

#### 在 DCU 上启用 FlagGems

DTK 自带 Triton（`hcu` 后端）和 FlagGems，其 `hygon` vendor 声明的
`device_name="cuda"` 正好契合 boxing 路线。两者都装在 DTK 的系统解释器里，因此
直接把环境指过去即可，不要装 PyPI 上面向 NVIDIA 的 wheel：

```bash
pip install pyyaml sqlalchemy          # flag_gems 依赖
mkdir -p /path/to/gems_path && cd /path/to/gems_path
ln -s /usr/local/lib/python3.10/dist-packages/triton .
ln -s /usr/local/lib/python3.10/dist-packages/flag_gems .

ACCELERATOR=dcu FLAGGEMS_PYTHON=1 pip install --no-build-isolation -e .

export PYTHONPATH=/path/to/gems_path
export TRITON_BACKENDS_IN_TREE=1       # 该安装没有 dist-info，
                                       # 基于 entry-point 的后端发现拿不到任何后端
export FLAGOS_USE_FLAGGEMS=1
pytest tests/integration/ops -q        # 无需再屏蔽 marker
```

DCU 构建会自动设置 `GEMS_VENDOR=hygon`，无需再手动导出。这一点不只影响 FlagGems：
`GEMS_VENDOR` 同时决定通信 profile（见 `torch_fl/comm/process_group.py`），而 DCU
属于 CUDA-ABI vendor，其 `ProcessGroupNCCL` 底层就是 RCCL。只有需要覆盖时才自行导出。

DCU 构建会把 `ACCELERATOR=dcu` 记录到 `torch_fl/_build_config.py`，因此运行时只需
`FLAGOS_USE_FLAGGEMS=1` 就会选中 `backends_dcu_flaggems.conf`，不必重新导出
`ACCELERATOR`。该配置即 `backends_flaggems.conf`，只是把 `hcu` Triton 编译不了或
跑不通的算子退回 cuda boxing 算子：`silu_backward`（`tl.math.div_rn` 缺少
`create_precise_divf` lowering）与 `slice_backward`（单独跑结果正确，但其梯度喂给
MIOpen 的 `convolution_backward` 会触发硬件 VMFault）。可用
`FLAGOS_OP_<name>=flagos_python|cuda` 按算子覆盖。

#### DCU 多卡

开着 FlagGems 也可以多卡，走 FlagCX 或 RCCL 回退路径均可，无需额外配置：自动设置的
`GEMS_VENDOR=hygon` 会把 `ProcessGroupFlagOS` 路由到 CUDA-ABI profile（零拷贝
`_flagos_to_cuda_view` + `ProcessGroupNCCL`，在 DTK 上底层即 RCCL：
`dist.is_nccl_available()` 为 `True`，`torch.cuda.nccl.version()` 返回
`(2, 22, 3)`）。

```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1     # 不设置时 RCCL 会告警；
                                       # 影响多卡吞吐与稳定性
python tests/manual/test_flagos_dist_live.py --world-size 2
```

这里的前提是工厂算子必须遵守传入的 `device` index：每个 rank>0 的 worker 都通过工厂
算子建张量，若输出分配在 device 0 而 Triton kernel 在 device N 上启动，就是一次跨设备
写入，会直接把 GPU 打挂。参见 `tests/integration/test_factory_device_index.py`。

### 从源码安装（燧原 GCU 平台）

燧原 GCU 走的是**原生算子路线**（与 Ascend 一致），而不是 CUDA boxing：
TopsRider 软件栈里没有 CUDA runtime 可以 box，厂商的 `torch-gcu` wheel 自己要占用
`PrivateUse1`，无法与 `torch_fl` 共存。因此本插件直接链接厂商库：

- `libtopsrt.so`（tops runtime）承载设备 / 内存 / 流层。
- `libtopsaten.so`（ATen 风格算子库）承载计算算子。它的调用形态是一次直调，
  没有 aclnn 那样的 workspace/executor 两段式。

```bash
# 上游 CPU 版 torch 即可，不需要厂商 torch。
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

ACCELERATOR=gcu pip install --no-build-isolation -vvv -e .
```

算子由 `scripts/codegen_gcu.py` 生成。它会把每个算子与 `libtopsaten.so` 中实际存在的
`topsaten::topsatenXxx`（demangle 后）符号比对，缺失的直接跳过，因此 codegen 不可能
生成当前 SDK 里没有的调用。

**GCU 平台注意事项：**

- **没有 topsaten kernel 的算子完全不在 `PrivateUse1` 上注册**，因此会落到
  `cpu_fallback` 而不是报错。要扩大覆盖面，只需扩充 `scripts/codegen_gcu.py`
  里的 `OPS` 表。
- **topsaten 没有 int64 kernel**：任何 `I64` 操作数都会返回 `NOT_SUPPORT`。
  因此每个生成的 kernel 都会先判断 dtype，int64 情况下在 CPU 上算再拷回，
  保证索引、mask、计数器这类张量可用。它同样拒绝 rank-0 形状，所以 0 维张量
  会按 1 元素向量来描述。
- **tops 设备指针是按设备绑定的**（没有统一寻址）：指针只在*当前*设备上有效，
  所以 allocator 和每个 kernel 都会先切换设备，默认流也是每设备一条。
- **带原生 kernel 的构建会安装一个 `lib/flagos_platform` 标记文件**，让
  `torch_fl` 选用 `backends_gcu.conf`，而不是默认的 CUDA 路由。
- 用 `-DGCU_KERNEL=OFF` 可以完全跳过 topsaten；此时 runtime 仍可用，
  所有计算回落到 CPU。

### 构建环境变量

| 变量 | 说明 |
|------|------|
| `ACCELERATOR` | 硬件平台：`cuda`（默认）、`metax`、`ascend`、`tsingmicro`、`dcu` 或 `gcu` |
| `FLAGOS_BUILD_JOBS` | 原生库并行编译线程数（默认 CPU 核数）；日志过长可设 `1` |
| `CUDA_HOME` | CUDA toolkit 路径 |
| `DTK_ROOT` | 海光 DTK 路径（依次回退到 `ROCM_PATH`、`/opt/dtk`；DCU 构建必需） |
| `TOPS_HOME` | 燧原 TopsRider SDK 路径（默认 `/opt/tops`；GCU 构建必需） |
| `METAX_PATH` | MetaX SDK 路径（默认 `/opt/maca`，metax 构建必需） |
| `METAX_ARCH` / `METAX_MXCC` | 可选：GPU 架构或 mxcc/cucc 编译器路径 |
| `METAX_KERNEL` | 启用 MetaX C++ kernel 构建（`ON`/`OFF`；`ACCELERATOR=metax` 时自动开启） |
| `ASCEND_HOME` | CANN toolkit 路径（默认 `/usr/local/Ascend/ascend-toolkit/latest`） |
| `FLAGGEMS_DIR` | FlagGems C++ 库路径（启用低开销 C++ dispatch） |
| `FLAGGEMS_KERNEL` | 启用 FlagGems C++ kernel 封装（`ON`/`OFF`，默认 `ON`；Ascend 设为 `0`） |
| `FLAGGEMS_PYTHON` | 启用 FlagGems Python kernel 封装（`ON`/`OFF`，默认 `OFF`；设为 `1` 启用） |
| `CUDA_KERNEL` | 启用 CUDA kernel 构建（`ON`/`OFF`，默认 `ON`；Ascend 设为 `0`） |
| `ASCEND_KERNEL` | 启用 Ascend kernel 构建（`ON`/`OFF`，默认 `OFF`；Ascend 设为 `1`） |
| `GCU_KERNEL` | 启用燧原 GCU topsaten kernel 构建（`ON`/`OFF`；`ACCELERATOR=gcu` 时自动开启） |

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
export FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_metax_flagos_py.conf
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

默认路径 `torch_fl/configs/backends.conf`，可通过 `FLAGOS_BACKEND_CONFIG` 环境变量覆盖：

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
| `torch_fl/configs/backends_metax.conf` | 所列算子全部 → `metax` C++ kernel。pytest 检测到 MetaX（`/dev/mxcd`）且未设置 `FLAGOS_BACKEND_CONFIG` 时自动选用。 |
| `torch_fl/configs/backends_metax_flagos_py.conf` | **集成测试推荐。** 混合路由：多数计算算子 → `flagos_python`；将 Triton 不兼容算子（`mm`/`bmm`/`mean.dim`）以及分配/工厂算子（`zeros`、`scalar_tensor`、`embedding` 等）保留在 `metax`。 |

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
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_flagos_py.conf \
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
export FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_metax_flagos_py.conf
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
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_metax.conf \
  FLAGOS_DISABLE_FLAGGEMS_PY=1 \
  pytest tests/integration/test_ops.py -v
```

未设置 `FLAGOS_BACKEND_CONFIG` 时，`tests/integration/conftest.py` 会在 MetaX 硬件上自动选择 `torch_fl/configs/backends_metax.conf`。

### Ascend 平台

```bash
# 算子测试
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf \
  pytest tests/integration/ops/ -v -m "anyplatform or ascend"

# Qwen3 推理测试
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf \
  pytest tests/integration/test_qwen3_infer.py -v -s

# Qwen3 训练测试（单卡）
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf \
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
