# Ascend NPU 落地方案与算子覆盖计划

> 起草日期：2026-07-20
> 机器：Ascend 910（8×，CANN 9.0.0，aarch64）
> 场景目标：**推理 + 训练都要**
> 主手段：**aclnn codegen 优先**（辅以 FlagGems/triton-ascend 与 CPU 兜底）

## 0. 结论先行

- CUDA 的"外挂 `libtorch_cuda.so` 零成本兜底"路线**在 NPU 上不成立**（torch_npu 与 flagos 同占 PrivateUse1 key，详见 [cpu_torch_external_libtorch_npu.md](cpu_torch_external_libtorch_npu.md)）。NPU 必须**自己出算子**。
- NPU 出算子的最优底座是 **CANN aclnn（`libopapi.so`）公开 C ABI**，现有 `csrc/aten/backends/ascend/`（33 个手写算子）已验证这条路可跑通。
- **决定性现状问题**：#10/`285f52c` 的 codegen 清理把 `csrc/aten/*.h` 的**逐算子头文件删掉、统一进 `csrc/aten/generated/ops.h`**，但 33 个手写 ascend `.cc` 仍 `#include "../../mm.h"` 这类**已不存在的头**。**当前 main 上 Ascend 后端无法编译**（这是任何 NPU 工作的第一道门槛）。
- 可行性已用**独立原型**在真机验证：`aclnn<Op>GetWorkspaceSize + aclnn<Op>` 两段式调用 + 裸 NPU 存储构造 `aclTensor`，`aclnnSqrt` 计算结果与 CPU 参考一致（`max_err=2.85e-07`）。见 §4。

## 1. 为什么是 aclnn codegen

CUDA 侧 `scripts/codegen_ops.py` 从 `native_functions.yaml` 批量生成 3429 个 boxing 兜底（`cuda_kernels.cc`），把 flagos 张量改元数据后转调 `at::xxx`（native CUDA kernel）。

Ascend 没有"独立 key 可 box 过去"，所以对应物不是 boxing，而是**批量生成 aclnn 调用胶水**：

```
aten op (schema)  ──codegen──▶  KernelAscend(...) { EXEC_ASCEND_CMD(aclnn<Op>, ...); }
                                └─ 注册进 flagos 内部 Dispatcher 的 kAscend 槽
```

aclnn 命名与调用高度规律：
- 命名：`aclnn` + 驼峰算子名（`aclnnMm` / `aclnnAdd` / `aclnnCos` / `aclnnSqrt` …）。
- 调用：统一两段式 `xxxGetWorkspaceSize(inputs..., out, &ws, &exec)` + `xxx(ws_addr, ws, exec, stream)`——已被 `EXEC_ASCEND_CMD`（`op_api_common.h`）抽象。
- 因此 elementwise / 一元数学 / 部分 reduce / matmul 类算子可由"`aten→aclnn` 映射表 + 类别模板"批量生成。

## 2. 现有资产盘点

| 资产 | 位置 | 状态 |
|---|---|---|
| aclnn 调用抽象 | `csrc/aten/backends/ascend/op_api_common.h`（`EXEC_ASCEND_CMD` / `AclTensorWrapper` / `AclScalarWrapper` / dtype 映射） | ✅ 可用 |
| 输出张量分配 | `op_preparation.h`（`apply_tensor_without_format` = `at::empty(device=PrivateUse1)`） | ✅ 可用 |
| 内部 Dispatcher | `csrc/aten/dispatcher.h`（`REGISTER_IMPL_TO_DISPATCHER(..., Backend::kAscend, ...)`） | ✅ 可用 |
| 手写算子 | `backends/ascend/*.cc`（33 个：mm/bmm/add/mul/cat/embedding/softmax/sum/nll_loss/index/…） | ⚠️ 头文件失效，需修 |
| 后端选择配置 | `torch_fl/backends_ascend.conf`（逐 op `flaggems\|ascend`） | ✅ 可用 |
| codegen 框架 | `scripts/codegen_ops.py` + `generated/name_map.json`（权威符号命名源） | ✅ 可复用其骨架 |
| 运行时（stream/allocator/device） | `csrc/runtime/accelerator/ascend/` | ✅ 已有 |

## 3. 方案（分层，按 conf 逐 op 选后端）

三层能力，`backends_ascend.conf` 决定每个 op 走哪层，覆盖不到的自动 CPU fallback：

1. **aclnn codegen（主）**——覆盖规律性强的算子（elementwise、一元数学、reduce、matmul 家族）。目标把手写的 33 个扩到上百个。
2. **FlagGems / triton-ascend（辅）**——融合算子、triton 能编过且更快的热点（已有 `_patch_flaggems_codegen_config` + `patch_triton_ascend.py` 基建）。
3. **CPU fallback（兜底）**——长尾/不常用算子，显式标注为已知性能点。

### 落地顺序

- **P0（阻塞项）：修复 Ascend 后端可编译。** 解决 33 个 `.cc` 引用的失效逐算子头。两个方向择一：
  - (a) codegen 为 ascend 也产出逐算子头（复活 `csrc/aten/*.h`）；或
  - (b) 改这些 `.cc` 统一 `#include "generated/ops.h"`（更契合 #10 后的单头结构，推荐）。
  - 先把 Ascend 后端在当前 main 上重新编过、`import torch_fl` 通、33 个算子回归通过，作为基线。
- **P1：aclnn codegen MVP。** 先覆盖"一输入一输出 elementwise/一元数学"这一最规律类别（sqrt/exp/reciprocal/sigmoid/tanh/floor/ceil/sign/gelu/…），建 `aten→aclnn` 映射表 + 一个类别模板，生成到 `backends/ascend/generated/`。此类别 `AclTensorWrapper(in)/(out)` + `EXEC_ASCEND_CMD(aclnn<Op>, in, out)` 即可，风险最低。
- **P2：扩类别。** 二元（add/mul/sub/div，处理 broadcast+alpha+dtype 提升，参考现有手写 `add.cc`）、reduce（sum/mean/max，处理 dim/keepdim）、matmul（mm/bmm，cube_math_type）。长尾 aclnn 名不规律或需特殊参数的进 skip 列表走 fallback。
- **P3：训练算子。** backward 系列（silu_backward/embedding_dense_backward/nll_loss_backward 已手写，补全 relu/gelu/norm 等），优化器 foreach 类评估 aclnnForeach* 覆盖度。

### 与 CUDA codegen 的关系

- 复用 `codegen_ops.py` 的 schema 解析（`native_functions.yaml` → 签名/类别/`fn_type`/`dispatcher`）与 `name_map.json` 命名权威。
- **新增** ascend 专属发射器：不发 boxing 体，发 aclnn 体；输入来源是 `aten→aclnn` 映射表（新文件，如 `torch_fl/ascend_aclnn_map.json`）而非 `backends_cuda.conf`。
- 生成物落 `csrc/aten/backends/ascend/generated/`，由 `csrc/CMakeLists.txt` 的 `ASCEND_KERNEL` glob 纳入。

## 4. 可行性验证（已在真机通过）

独立原型（不依赖 torch_fl 构建，规避 P0 阻塞）证明 codegen 将要发射的**内核体形状**在真机可算：

- 源码：[`docs/ascend_aclnn_codegen_prototype.cc`](ascend_aclnn_codegen_prototype.cc)
- 构建运行：[`docs/build_ascend_prototype.sh`](build_ascend_prototype.sh)
- 做法：裸 `aclrtMalloc` 显存 → `aclCreateTensor`（同 `AclTensorWrapper`）→ 两段式 `aclnnSqrtGetWorkspaceSize` + `aclnnSqrt`（同 `EXEC_ASCEND_CMD`）→ 拷回校验。
- 结果：

  ```
  aclnnSqrt sample: in[3]=4.0 out[3]=2.000000 (ref=2.000000)  max_err=2.850e-07
  PASS: aclnnSqrt on NPU matches CPU reference
  ```

**结论**：aclnn 两段式 + 裸存储 aclTensor 的 codegen 体形状**成立**。剩下的是工程化（映射表、类别模板、P0 编译修复），无底层不确定性。

## 5. 风险与代价

- ⚠️ **P0 编译修复是硬前置**，否则任何 NPU 算子都无法验证。
- ⚠️ aclnn 名/签名的长尾不规律：codegen 吃掉 60–80% 规律算子，尾部仍需手写或 fallback，需维护 skip 列表（类比 `codegen_skip_ops.txt`）。
- ⚠️ 二元/reduce 的 broadcast、dtype 提升、alpha、dim/keepdim 语义需在模板里正确处理（现有手写 `add.cc`/`sum.cc`/`mean.cc` 是参考样本）。
- ⚠️ 强绑 CANN 版本（aclnn 接口随 CANN 演进），换 CANN 需回归。

## 6. 下一步（本轮之后）

1. 定 P0 修复方向（推荐 (b) 统一 `generated/ops.h`），跑通 Ascend 基线。
2. 建 `ascend_aclnn_map.json` + P1 一元 elementwise 类别模板，生成并回归。
3. 逐步 P2/P3 扩类别，长尾进 skip 走 fallback。
