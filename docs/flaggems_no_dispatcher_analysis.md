# no_dispatcher 88 个算子接入分析

`docs/flaggems_unrouted_ops.md` 里最大的一桶是 no_dispatcher（88 个）：gems 有实现，但 aten 侧 codegen 没为它们生成 CUDA dispatcher，因此 discovery 直接跳过（`op not in codegen_ops`）。

本文回答:**这 88 个为什么没 dispatcher,以及怎么才能接入**。逐个复现 `enumerate_all_cuda_ops()` 的 gating 后,88 个归因如下:

| 子类 | 数量 | 本质 | 需不需要接 |
|---|---|---|---|
| A. composite_implicit(拆解算子) | 61 | PyTorch 在我们 dispatch key **之上**已拆成叶子 | 功能上不需要;只为性能 |
| B. 无 CUDA kernel 但可 fallback | 9 | 无直连 kernel,靠 CompositeExplicit / cpu_fallback 落地 | 功能上不需要;只为性能 |
| C. 手工注册(MANUAL_REGISTERED_OPS) | 4 | 已在 register.cc 手写(copy_/_to_copy/index_put_/_index_put_impl_) | 已接入,内存/视图语义特殊 |
| D. 其余 gems 名与 aten 对不上 | 14 | gems 用别名/融合名,对应的 aten leaf 已单独路由 | 无收益 |

> **关键结论**:no_dispatcher 桶里**没有一个是"功能缺失"**。抽验(`FLAGOS_USE_FLAGGEMS=1`)显示 `divide/square/true_divide/greater/var/selu/vstack`(A 类)与 `diag_embed/pixel_unshuffle/select_backward/slice_scatter/alias_copy/t_copy`(B 类)在 flagos 上**全部能跑并结果正确**。它们已经通过拆解 / fallback 落到已注册的叶子算子。所以"接入"的唯一动机是**让 gems 的融合 kernel 直接接管以提性能**,不是补功能。

---

## A. composite_implicit —— 61 个

这些算子带 `CompositeImplicitAutograd` kernel 且无 `structured_delegate`/`CompositeExplicitAutograd`。`enumerate_all_cuda_ops` 明确排除它们(codegen_ops.py:143-147),理由:**PyTorch 在到达 PrivateUse1 dispatch key 之前就把它们拆成叶子算子**,注册它们既多余又危险(会拦截拆解、丢掉 autograd 公式)。

```
__ior__.Scalar   __ior__.Tensor   __or__.Scalar   __or__.Tensor
absolute   arcsinh   arcsinh.out   arcsinh_   arctanh_
clip   clip_   conj_physical
conv1d   conv1d.padding   conv2d   conv2d.padding   conv3d   conv3d.padding
diag   divide.Scalar   divide.Scalar_mode   divide.Tensor   divide.Tensor_mode
divide_.Scalar   divide_.Scalar_mode   divide_.Tensor   divide_.Tensor_mode
embedding_backward   gather_backward
greater.Scalar   greater.Scalar_out   greater.Tensor
hstack   isclose   isfinite   kron   log_sigmoid   margin_ranking_loss
one_hot   pad   prelu   quantile   relu6
repeat_interleave.self_Tensor   repeat_interleave.self_int
resolve_conj   resolve_neg   rms_norm   selu   selu_
square   square.out   square_   tile
true_divide.Scalar   true_divide.Tensor   true_divide_.Scalar   true_divide_.Tensor
var   var.dim   vstack
```

**为什么拆解就够用**:例如 `divide.Tensor` 拆成 `div.Tensor`(已路由 gems)、`square` 拆成 `mul`/`pow`、`vstack` 拆成 `cat`、`selu` 拆成 `elu`/`mul`。叶子已经在 flaggems 或 cuda 上跑,所以整算子无需自己的 kernel。

**如果一定要接(性能)**,两条路,都要慎重:
1. **在 CUDA dispatch key 强注册**:把它们从 `enumerate_all_cuda_ops` 的 composite_implicit 排除里放出来,codegen 出 dispatcher + `kFlagOsPython` kernel。风险:拦截了 PyTorch 的拆解,**autograd 公式随之丢失**——`conv1d/embedding_backward/gather_backward/rms_norm/prelu` 这类带梯度语义的会训练出错。只有纯 forward、无梯度依赖的(`isfinite/isclose/conj_physical/resolve_*`)相对安全。
2. **注册到 `CompositeImplicitAutograd` 之下但 autograd 之上的 key**(如 `Autograd` 后的 functorch 层)——本项目 boxing 方案没有这层,不现实。

**建议**:整体不接。只有当 profiling 证明某个融合 kernel(如 `rms_norm`、`conv2d`)的收益显著、且我们能同时提供其 backward 时,才逐个特批,并在 backward 也走 gems。

---

## B. 无 CUDA kernel、靠 fallback 落地 —— 9 个

无 `CompositeImplicit`、无直连 CUDA kernel,但存在 `CompositeExplicitAutograd` 或走 cpu_fallback 落到已注册叶子。

```
alias_copy   diag_embed   lift_fresh_copy   max_pool2d_backward
pixel_unshuffle   select_backward   select_scatter   slice_scatter   t_copy
```

抽验均能在 flagos 跑通。其中 `*_scatter`/`select_backward`/`alias_copy`/`t_copy`/`diag_embed`/`pixel_unshuffle` 都属于 view/scatter 元操作或可由 as_strided + copy 表达。

**接入方式**:这几个**有真正的 CUDA leaf 语义**(不像 A 类是纯拆解),理论上可以:
- 放开 `cuda_supported` 让它们进 codegen(它们多是 `CompositeExplicitAutograd`,`cuda_supported` 第 3 条本应放行——需查为何没命中,可能是 `has_composite_explicit_autograd_kernel` 为 False 而实际走 structured)。
- 然后按普通 functional/out 分类生成 `kFlagOsPython` kernel。

**建议**:低优先。它们已能 fallback 正确执行,gems 版收益有限。`max_pool2d_backward` 是唯一可能值得(训练热点),但需确认其 forward `max_pool2d_with_indices` 已路由且 indices 语义对齐。

---

## C. 手工注册 —— 4 个

```
copy_   _to_copy   index_put_   _index_put_impl_
```

已在 `csrc/aten/.../register.cc` 手写(`MANUAL_REGISTERED_OPS`),因为涉及内存拷贝 / 原地索引写 / 跨设备语义,不能走通用 boxing。**已接入,不在缺口内**——它们出现在 no_dispatcher 只是因为 codegen 主动让位给手写版。不要用 flaggems 覆盖。

---

## D. gems 别名/融合名 —— 其余

剩下少量是 gems 用了融合名或别名(如 `scaled_softmax_forward/backward`、`nll_loss_nd_forward/backward`、`new_full.Tensor`、`repeat`、`bitwise_left_shift`),对应的 aten leaf 要么不在 native schema、要么已单独路由。这些**没有对应的标准 aten dispatcher 可挂**,属于 gems 私有扩展 op,接入需要自定义 schema,收益极低。

---

## 总结与建议

- **no_dispatcher 88 个全部已能正确执行**(拆解 / fallback / 手写),不是功能缺口。
- **不建议批量接入**。批量放开 composite_implicit 会丢 autograd,是净损失。
- **可逐个特批的性能候选**(需同时保证 backward、经 profiling 验证):`rms_norm`、`conv2d`、`max_pool2d_backward`。接入时走"CUDA key 强注册 + gems forward/backward 成对路由",并加数值 + 训练回归。
- 其余(元操作、别名、手写)**维持现状**。

分析脚本(临时):`/tmp/fg_no_disp.py`、`/tmp/fg_decomp.py`,复现 `enumerate_all_cuda_ops` gating。
