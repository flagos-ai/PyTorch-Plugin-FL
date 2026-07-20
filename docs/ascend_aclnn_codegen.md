# Ascend aclnn 算子 codegen 方案

> 目标：把 `csrc/aten/backends/ascend/` 的手写 aclnn 内核，用代码生成的方式批量扩到全量，
> 低维护成本地覆盖推理/训练主干算子。
>
> 状态：设计 + unary 类别原型（真机验证通过）。日期 2026-07-20，torch 2.11 分支。

## 1. 为什么不能照搬 CUDA codegen

CUDA 侧 `scripts/codegen_ops.py` 生成 `generated/cuda_kernels.cc`，内核体就一行
`at::op(args)` —— 靠 `DeviceBoxingGuard` 把 flagos(PrivateUse1)张量的 device 元数据
改成 CUDA，直接复用 PyTorch 已注册的 CUDA kernel。这条捷径在 Ascend 上不成立：

- torch_npu 与 flagos 同占 PrivateUse1，无独立 key 可 box（见 [[ascend-libtorch-npu-fallback-fails]]、
  [[ascend-route1-intercept-fails]]，均已实测封死）。
- 所以 Ascend 内核体必须**自己调 CANN aclnn**（`libopapi.so` 公开 C ABI），
  这意味着每个算子要手动：把参数包成 `AclTensorWrapper`/`AclScalarWrapper`/`AclIntArrayWrapper`，
  **自己分配输出、推形状/ dtype**，再走两段式 `GetWorkspaceSize` + `Execute`。

| | CUDA codegen | aclnn codegen |
|---|---|---|
| 内核体 | `at::op(args)` 一行 | 参数编组 + 输出分配 + `EXEC_ASCEND_CMD(aclnn<Name>, ...)` |
| 信息来源 | 全在 aten schema | schema **不含** aclnn API 名 / 参数编组规则 |
| 覆盖策略 | 枚举全部 CUDA 算子 | **按类别 + 映射表**，逐类扩 |

关键洞察：aclnn 调用范式高度统一（`EXEC_ASCEND_CMD` 已抽象两段式），真正的差异只在
"参数怎么编组、输出怎么分配"——而这些**在同一类别内是完全一致的**。所以 codegen 以
**category（类别）** 为核心。

## 2. 与现有基建的关系

- **dispatcher 声明复用**：`generated/ops.h` 里已有全量 `XxxFn` typedef + `DECLARE_DISPATCHER`，
  `ops.cc` 里已有 `ADD_IMPL_TO_DISPATCHER`，`register.inc` 里已有 `m.impl("op", WrapperXxx)`
  把 aten 算子绑到 `xxx_dispatcher`。**ascend codegen 不重复声明任何 dispatcher**，
  只生成 `REGISTER_IMPL_TO_DISPATCHER(XxxFn, xxx_dispatcher, Backend::kAscend, XxxKernelAscend)`
  把内核挂到已存在的 dispatcher 的 `kAscend` 槽。
- **符号名一致**：生成器复用 `codegen_ops.py:schema_to_cpp_name()`，保证 `XxxFn`/`xxx_dispatcher`
  与 CUDA codegen 完全对齐（否则 link 不上）。
- **运行时选择**：`torch_fl/backends_ascend.conf` 里 `op = ascend` 的行，让 `GetBackendForOp`
  在运行时把该 op 路由到 `kAscend` 槽。codegen 会顺带把生成的算子写进这个 conf。

## 3. 落点与构建

生成文件：`csrc/aten/backends/ascend/generated/ascend_kernels.cc`

- 该路径在 `csrc/CMakeLists.txt` 已被非 ascend 构建自动排除
  （`if(NOT ASCEND_KERNEL) EXCLUDE ".*/aten/backends/ascend/.*"`），无需新增 CMake 规则。
- include：
  ```cpp
  #include "../../../generated/ops.h"   // Fn typedef + DECLARE_DISPATCHER
  #include "../op_preparation.h"        // OpPreparation::apply_tensor_without_format
  #include "../op_api_common.h"         // AclTensorWrapper / EXEC_ASCEND_CMD
  ```
- 与手写内核**互斥**：一个 op 要么手写、要么 codegen，不能同时注册 `kAscend`（重复注册报错）。
  codegen 读一份 skip 名单排除已手写的 op。

## 4. 类别体系（逐类扩）

| category | 判据 | 输出形状 / dtype | 内核体模板 |
|---|---|---|---|
| `unary` | 1 个 Tensor 入、Tensor 出、无其它张量/标量 | = 输入 | `aclnn<Name>(self, out)` |
| `binary` | 2 个 Tensor 入 | broadcast(self, other) | `aclnn<Name>(self, other, out)`（可选 alpha） |
| `binary_scalar` | Tensor + Scalar | = 输入 | `aclnn<Name>s(self, scalar, out)` |
| `reduce` | Tensor + dim + keepdim | 按 dim 缩 | 需 `AclIntArrayWrapper`，长尾 |
| `matmul` 等 | 手写保留 | — | — |

本轮原型只实现 **unary**，把 sqrt/exp/tanh/sigmoid/reciprocal/log/floor/ceil 等
尚未手写的一元算子一次性接入，证明 codegen 净增覆盖。

## 5. aclnn 命名派生

- 默认：`op_name` snake_case → `aclnn` + PascalCase（`sqrt`→`aclnnSqrt`，`floor`→`aclnnFloor`）。
- 不规则：显式覆盖表（`bmm`→`BatchMatMul`、`sum`→`ReduceSum`、`where`→`SWhere`、
  `bitwise_and`→`BitwiseAndTensor` 等）。实测 unary 候选中 33/38 可直接派生。
- 生成前用 `nm libopapi.so` / aclnn 头存在性校验，派生不出或库里没有的 op 直接跳过并告警。

## 6. 生成器 `scripts/codegen_ascend.py`

输入：
- `torch_fl/backends_ascend.conf`（哪些 op 要 ascend 后端）或 `--category unary` 枚举模式
- torchgen 的 `native_functions.yaml`（取 schema、复用 `schema_to_cpp_name`）
- aclnn 覆盖表（内嵌 dict）+ `libopapi.so` 符号校验

输出：
- `csrc/aten/backends/ascend/generated/ascend_kernels.cc`
- 顺带把新覆盖的 op 追加到 `backends_ascend.conf`

## 7. 验证闭环

`ACCELERATOR=ascend ASCEND_KERNEL=1 FLAGGEMS_PYTHON=1 ...` 构建，
`FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf`，逐 op 与 CPU 对拍。

## 8. 相关

见 `docs/ascend_npu_plan.md`（总纲）、`docs/ascend_aclnn_codegen_prototype.cc`（裸 aclnn 原型）。
记忆：[[ascend-aclnn-codegen-plan]]、[[ascend-backend-broken-on-2.11]]。
