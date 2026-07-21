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

已实现 40 个类别，共 103 个算子（真机全部与 CPU 对拍通过）：

| category | 判据 | 输出形状 / dtype | 内核体模板 |
|---|---|---|---|
| `unary` | 1 个 Tensor 入、Tensor 出 | = 输入 | `aclnn<Name>(self, out)` |
| `unary_bool` | 1 个 Tensor 入、判定 | = 输入 shape，**bool 出** | `aclnn<Name>(self, out)` |
| `unary_scalar` | Tensor + Scalar | = 输入 | `aclnn<Name>(self, s, out)` |
| `unary_two_scalar` | Tensor + Scalar×2 | = 输入 | `aclnn<Name>(self, s1, s2, out)` |
| `unary_int` | Tensor + `int64_t` | = 输入 | `aclnn<Name>(self, i, out)` |
| `unary_dims` | Tensor + `IntArrayRef` | = 输入 | `aclnn<Name>(self, dims, out)` |
| `binary` | 2 个 Tensor 入 | broadcast(self, other)，= self dtype | `aclnn<Name>(self, other, out)` |
| `binary_alpha` | 2 个 Tensor + Scalar alpha | broadcast | `aclnn<Name>(self, other, alpha, out)` |
| `binary_cmp` | 2 个 Tensor 入、比较 | broadcast，**bool 出** | `aclnn<Name>(self, other, out)` |
| `binary_scalar_alpha` | Tensor + Scalar other + Scalar alpha | = 输入 | `aclnn<Name>s(self, other, alpha, out)` |
| `binary_scalar_cmp` | Tensor + Scalar、比较 | = 输入 shape，**bool 出** | `aclnn<Name>(self, other, out)` |
| `addcmul` | self + t1 + t2 + Scalar value | broadcast(3) | `aclnn<Name>(self, t1, t2, value, out)` |
| `pow_scalar_tensor` | Scalar self + Tensor exponent | = exponent | `aclnn<Name>(self, exp, out)` |
| `reduce_dims` | Tensor + `IntArrayRef dim` + keepdim | 按 dim 缩 | `aclnn<Name>(self, dim, keepdim, out)` |
| `reduce_dim_bool` | Tensor + `int64_t dim` + keepdim | 按单 dim 缩，**bool 出** | `aclnn<Name>(self, dim_list, keepdim, out)` |
| `reduce_max_dim` | Tensor + `int64_t dim` + keepdim | **tuple(values, indices)** | `aclnn<Name>(self, dim, keepdim, val, idx)` |
| `cumsum` | Tensor + `int64_t dim` + optional dtype | = 输入 shape（扫描） | `aclnn<Name>(self, dim, dtype, out)` |
| `cumprod` | 同 cumsum，但 dim 以 `aclScalar*` 传 | = 输入 shape（扫描） | `aclnn<Name>(self, &dim, dtype, out)` |
| `act_backward` | grad_output + output | = output | `aclnn<Name>(grad, output, grad_in)` |
| `threshold_backward` | grad_output + self + Scalar threshold | = self | `aclnn<Name>(grad, self, thr, grad_in)` |
| `elu` | Tensor + Scalar×3（alpha/scale/input_scale） | = 输入 | `aclnn<Name>(self, a, s, is, out)` |
| `loss` | self + target + `int64 reduction` | None→输入 / Mean·Sum→标量 | `aclnn<Name>(self, target, reduction, out)` |
| `cummax_cummin` | Tensor + `int64_t dim` | **tuple(values, indices)**，= 输入 shape | `aclnn<Name>(self, dim, val, idx)` |
| `aminmax` | Tensor + optional dim + keepdim | **tuple(min, max)** | `aclnn<Name>(self, dim_list, keepdim, min, max)` |
| `prod` | Tensor + optional dtype | 标量 | `aclnn<Name>(self, dtype, out)` |
| `gemm_addmm` | self + mat1 + mat2 + beta + alpha | (m1.rows, m2.cols) | `aclnn<Name>(self,m1,m2,beta,alpha,out,cubeMathType)` |
| `gemm_baddbmm` | self + batch1 + batch2 + beta + alpha | (b, b1.rows, b2.cols) | 同上（batched） |
| `mv` | self (n,m) + vec (m,) | (n,) | `aclnn<Name>(self, vec, out, cubeMathType)` |
| `dot` | self + tensor（均 1-D） | 标量 | `aclnn<Name>(self, tensor, out)` |
| `layer_norm` | input + normalized_shape + optional weight/bias + eps | **tuple(out, mean, rstd)**；out=输入，stat=前缀维+1 | `aclnn<Name>(input, ns, weight, bias, eps, out, mean, rstd)` |
| `group_norm` | input + optional weight/bias + N/C/HxW/group + eps | **tuple(out, mean, rstd)**；out=输入，stat=(N,group) | `aclnn<Name>(input, weight, bias, N, C, HxW, group, eps, out, mean, rstd)` |
| `gelu` | Tensor + `approximate` string_view | = 输入 | `aclnnGeluV2(self, approx_int, out)`（int64 0=none/1=tanh） |
| `gelu_backward` | grad_output + self + `approximate` | = self | `aclnnGeluBackwardV2(grad, self, approx_str, grad_in)`（char\* 字符串） |
| `log_softmax` | Tensor + `int64_t dim` + half_to_float | = 输入（half_to_float→float 出） | `aclnn<Name>(self, dim, out)` |
| `softmax_backward` | grad_output + output + `int64_t dim` + input_dtype | = grad_output shape，dtype=input_dtype | `aclnn<Name>(grad, output, dim, grad_in)` |
| `gemm_addmv` | self + mat(n,m) + vec(m) + beta + alpha | (n,) | `aclnn<Name>(self,mat,vec,ALPHA,BETA,out,cubeMathType)`（**alpha 在 beta 前**） |
| `gemm_addr` | self + vec1(n) + vec2(m) + beta + alpha | (n,m) 外积 | `aclnn<Name>(self,vec1,vec2,beta,alpha,out)`（无 cubeMathType） |
| `bce` | self + target + optional weight + int reduction | None→输入 / Mean·Sum→标量 | `aclnn<Name>(self,target,weight,reduction,out)` |
| `bce_backward` | grad_output + self + target + optional weight + reduction | = self | `aclnn<Name>(grad,self,target,weight,reduction,grad_in)` |
| `bce_logits` | self + target + optional weight + optional pos_weight + reduction | None→输入 / Mean·Sum→标量 | `aclnn<Name>(self,target,weight,posWeight,reduction,out)` |

- **unary（28）**：sqrt/exp/tanh/sigmoid/reciprocal/log/floor/ceil/erf/erfc/expm1/
  log2/log10/log1p/round/trunc/frac/sign/relu/cosh/sinh/asin/atan/asinh/acosh/atanh/
  logical_not/bitwise_not
- **unary_bool（1）**：isinf
- **unary_scalar（4）**：leaky_relu/clamp_min/clamp_max/fmod.Scalar
- **unary_two_scalar（2）**：softplus(beta,threshold)/threshold(threshold,value)
- **unary_int（2）**：tril/triu（diagonal 偏移）
- **unary_dims（1）**：flip
- **binary（9）**：div.Tensor/pow.Tensor_Tensor/atan2/maximum/minimum/bitwise_or/bitwise_xor/
  fmod.Tensor/floor_divide
- **binary_alpha（1）**：sub.Tensor
- **binary_cmp（8）**：eq/ne/gt/lt/ge.Tensor + logical_and/logical_or/logical_xor
- **binary_scalar_alpha（2）**：add.Scalar/sub.Scalar
- **binary_scalar_cmp（6）**：eq/ne/gt/lt/ge/le.Scalar
- **addcmul（2）**：addcmul/addcdiv
- **pow_scalar_tensor（1）**：pow.Scalar
- **reduce_dims（2）**：amax/amin（复用 `sum.cc` 的 dim 归一化 + 缩形状逻辑）
- **reduce_dim_bool（1）**：any.dim（单 dim 包成一元 list 传给 aclnn）
- **reduce_max_dim（2）**：max.dim/min.dim（tuple 返回 values+int64 indices）
- **cumsum（1）**：cumsum；**cumprod（1）**：cumprod
- **act_backward（2）**：tanh_backward/sigmoid_backward（训练用）
- **threshold_backward（1）**：threshold_backward（relu 反向，训练用）
- **unary_scalar 补充（3）**：celu/softshrink/hardshrink；**unary_two_scalar 补充（1）**：hardtanh
- **elu（1）**：elu
- **loss（1）**：mse_loss（reduction 决定标量/逐元素输出）
- **cummax_cummin（2）**：cummax/cummin（tuple 返回，同形状扫描）
- **aminmax（1）**：aminmax（tuple(min,max)，optional dim）
- **prod（1）**：prod（缩到标量）
- **gemm 家族（6）**：addmm/baddbmm（cube_math_type）/mv/dot/addmv（alpha,beta 顺序反）/addr（无 cube_math_type）
- **bce 家族（3）**：binary_cross_entropy（+ optional weight）/binary_cross_entropy_backward/binary_cross_entropy_with_logits（+ optional pos_weight）
- **layer_norm（1）**：native_layer_norm（tuple(out,mean,rstd)，transformer 主干）
- **group_norm（1）**：native_group_norm（tuple(out,mean,rstd)）
- **gelu（1）**：gelu（用 aclnnGeluV2 支持 none/tanh 两种近似；见下方"gelu 的 V2 坑"）
- **gelu_backward（1）**：gelu_backward（aclnnGeluBackwardV2，char\* approximate）
- **log_softmax（1）**：_log_softmax（照搬手写 softmax.cc 范式）
- **softmax_backward（2）**：_softmax_backward_data/_log_softmax_backward_data（训练用；aclnn 名去掉 aten 的 `_data` 后缀）

长尾未接（进后续或手写）：var/std.correction、norm.ScalarOpt_dim（correction/p 参数）、
argmax/argmin/logsumexp/isnan/masked_fill/remainder/relu6（无 aclnn 符号或需特殊派生）、
卷积/池化家族（各自 bespoke，需输出形状公式，专门批次）。
- **addbmm**：符号存在且能跑，但 hf32 cube 沿 batch 维累加把相对误差放大到 ~1e-2
  （单次 addmm 仅 ~1e-4）。留待允许 fp32 累加或降 cubeMathType 时再接。
- **native_batch_norm**：`aclnnBatchNorm` 对 2D (N,C) 输入正常，但 4D NCHW 输入返回
  `ACLNN_ERR_INNER_NULLPTR`(561103) —— GetWorkspaceSize 阶段就失败，疑似需要特定
  format 或改用 BatchNormV2/BatchNormReduce 组合。留待 conv/pool 专门批次一起做。

**关键坑（gelu 的 V2）**：`aclnnGelu`（v1）硬编码 **tanh** 近似，而 PyTorch 的 `gelu`
默认 `approximate="none"`（erf 形式，qwen3 等主干用这个）。直接用 v1 会让默认 gelu 静默
产生 ~4.5e-4 的系统性误差（不是精度抖动，是近似形式不同）。修复：改用 `aclnnGeluV2`
（`int64_t approximate`：0=none/1=tanh，int 变参安全）与 `aclnnGeluBackwardV2`
（`char* approximate` 字符串——指针传递也变参安全，不受下方 by-value float 坑影响）。

**关键坑（varargs float）**：`EXEC_ASCEND_CMD` 通过 `typedef int (*)(...)` 变参函数指针调用
aclnn。aarch64 上按值传 `float` 会走默认实参提升（float→double）+ 错误寄存器类，导致 aclnn
读到垃圾值。所有标量都以 `aclScalar*` 指针或 `int64_t` 传递是安全的；唯独 smooth_l1_loss 的
`float beta` 是按值传 float——实测 beta 恒为 0（退化成纯 L1）。故 smooth_l1_loss 暂留长尾，
需要按值 float 参数的 aclnn 都要走非变参的显式 dlsym 调用（参考手写 `le.cc`）。

### 二元类别的共享 prologue（关键坑）

所有 tensor-tensor 类别共用一段 prologue，做三件事：

```cpp
auto result_dtype = self.scalar_type();
// ① other 必须同时对齐 device 和 dtype——不只是 dtype！
auto other_c = other.is_privateuseone()
    ? (other.scalar_type() == result_dtype ? other : other.to(result_dtype))
    : other.to(self.options());           // CPU other → 搬到 flagos(NPU)
auto out_shape = at::infer_size(self.sizes(), other_c.sizes());
auto self_b  = self.expand(out_shape).contiguous();   // aclnn 不总是自己 broadcast
auto other_b = other_c.expand(out_shape).contiguous();
```

**踩过的坑**：`torch.sub(x, 3.0)` 这类"张量 + python 标量"在 PyTorch 里会把标量包成
**CPU 标量张量**并派发到 `aten::sub.Tensor`（不是 sub.Scalar）。若 prologue 只对齐 dtype
不对齐 device，CPU 存储会被 `AclTensorWrapper` 当成 NPU 显存地址读取 → 全 nan。
修复即上面 `other.to(self.options())`（与手写 `add.cc` 一致）。这一处同时修好了
sub/div/pow 等所有"标量走 Tensor overload"的路径。

## 5. aclnn 命名派生

- 默认：`op_name` snake_case → `aclnn` + PascalCase（`sqrt`→`aclnnSqrt`，`floor`→`aclnnFloor`）。
- 不规则：`OPS` dict 里逐 op 显式覆盖 stem（`eq.Tensor`→`EqTensor`、`div.Tensor`→`Div`、
  `pow.Tensor_Tensor`→`PowTensorTensor`、`add.Scalar`→`Adds` 等）。
- 生成前用 `nm -D libopapi.so` 校验 `aclnn<Name>` 与 `aclnn<Name>GetWorkspaceSize` 两个符号，
  缺任一即跳过并告警（例如 `square`/`isnan`/`isfinite` 因无 dispatcher 或无符号被自动排除）。

## 6. 生成器 `scripts/codegen_ascend.py`

结构：
- `OPS` dict：`schema op 名 → (category, aclnn-name override)`，是唯一的手工维护点。
- `SKIP` set：已手写 kAscend 的 op（abs/cos/add.Tensor/mul.Tensor/le.Tensor…），
  绝不重发（重复注册 kAscend 会在 import 时崩）。
- `CATEGORIES` dict：`category → 内核体模板字符串`，加新类别 = 加一个模板 + 一批 OPS 条目。
- 复用 `codegen_ops.py:schema_to_cpp_name` 保证 `XxxFn`/`xxx_dispatcher` 与 `ops.h` 完全对齐。

用法：`python scripts/codegen_ascend.py [--category unary] [--no-conf]`（默认 all）。

输出：
- `csrc/aten/backends/ascend/generated/ascend_kernels.cc`
- 幂等重写 `backends_ascend.conf` 末尾的 `# --- generated ---` 块（追加新覆盖的 op）

## 7. 验证闭环

`ACCELERATOR=ascend ASCEND_KERNEL=1 FLAGGEMS_PYTHON=1 CUDA_KERNEL=0 FLAGGEMS_KERNEL=0`
构建，`FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf`，逐 op 与 CPU 对拍。
51/51 通过（unary max_err≤4.4e-5，binary≤4.7e-6，比较类精确匹配）。

**注意**：编译时必须显式传全套 env（`ACCELERATOR=ascend …`），否则 setup.py 默认
`ACCELERATOR=cuda`，cmake 配置阶段就会失败。

## 8. 相关

见 `docs/ascend_npu_plan.md`（总纲）、`docs/ascend_aclnn_codegen_prototype.cc`（裸 aclnn 原型）。
记忆：[[ascend-aclnn-codegen-plan]]、[[ascend-backend-broken-on-2.11]]。
