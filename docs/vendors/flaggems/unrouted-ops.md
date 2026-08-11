# FlagGems 未接入算子清单

FlagGems `_FULL_CONFIG` 共 **433** 个算子,当前 **320 已路由**到 `flagos_python` 路径,**113 个未以自身名字进路由表**。本文按原因分桶列出,供后续逐批攻关。

> **2026-07 更新(本轮 +13,307 → 320)**:攻下 ③varargs 与部分 rng。
> - **varargs 一元 inplace(7)**:`asinh_`/`sinh_`/`log1p_`/`digamma_`/`sgn_`/`hardswish_`/`logit_` —— gems wrapper 是 `(*args,**kwargs)` 无法内省 arity,但 aten schema 是权威 arity。加 `_FLAGGEMS_ARITY_OVERRIDE` 显式白名单(仅实测跑通、数值正确、不丢参的简单 elementwise)绕过 npos 闸门。`logit_` npos=2(self+eps 均位置传,eps=None ok)。实测数值与 CPU 一致(maxdiff ≤ 1e-6)。
> - **rng(6)**:`rand`/`randn`(factory)、`rand_like`/`randn_like`(like_factory)、`randperm`(factory,本 torch 版本 schema 无 generator 参)、`multinomial`(新 `rng_dropgen` 类别,剥离尾部 `Generator?`)。阻塞点是 gems `philox_backend_seed_offset(increment)` 取空的 `torch.cuda.default_generators`(CPU-torch+cuda shim,len=0)→ IndexError。运行时在 `torch_fl/__init__.py._patch_flaggems_philox()` monkeypatch 该函数注入 fallback CUDA generator 一次性解锁。实测分布正确、连调结果不同(offset 推进)。
> - **维持排除**:`i0_`、`zero`、`zero.out`(gems kernel 硬断言 `tensor.is_cuda` / "Input tensor must be on a CUDA device",flagos 是 PrivateUse1 永不满足,`_FLAGGEMS_ARITY_OVERRIDE` 仅记录 arity,实际进 `FLAGGEMS_PYTHON_SKIP`);`normal_`/`normal.*`(gems 硬编码 `generator=None` 不透传,上游 bug)。

数据由 `discover_flaggems_ops()`(`scripts/codegen_ops.py`)的逐分支拒绝逻辑对账得出,分桶与实际 codegen 拒绝完全一致。

> **重要更正(2026-07 实测)**:①no_dispatcher(88)这一桶**绝大多数并非功能缺口**。实机在 flagos 上逐个探测(`FLAGOS_USE_FLAGGEMS=1`,55 个代表性 op),**54 个 PASS、0 个数值错、0 个真实崩溃**。原因是这些 op 属 `composite_implicit_autograd`,PyTorch 在 PrivateUse1 dispatch key **之上**就把它们分解成 leaf op(conv2d→convolution、divide→div、var→…),而那些 leaf 已经路由好了。给它们单独补 dispatcher 无益(dispatcher 永不命中,属死代码),甚至有害。**"未进路由表" ≠ "不能用"**。

| 桶 | 数量 | 一句话原因 |
|---|---|---|
| ① no_dispatcher | 88 | **多为设计使然,非缺口**:composite 分解到已路由 leaf,已能跑(见上方更正) |
| ② type_unsupported_kwarg | 13 | 参数类型通用 caller 表达不了(Generator?/Device?/Layout?/MemoryFormat?/Tensor?) |
| ③ varargs | 12 | gems 签名 `(*args, **kwargs)`,arity 无法内省 |
| ④ manual_skip | 12 | 运行期崩溃,手工排除(device assert / 必填 out / rng) |
| ⑤ name_mismatch | 2 | 尾部 aten 参名对不上 gems keyword-only 参名 |
| ⑥ 其余零散 | 5 | foreach / optlist / arity 重排陷阱 |
| **合计** | **132** | |

---

## ① no_dispatcher —— 88 个(实测:大多已通过 leaf 分解可用)

gems 有实现,aten 侧 codegen **没以该 op 名生成 dispatcher**。但 torchgen 分类 + 实机验证表明,这 88 个按"该不该补 dispatcher"分成四类:

| 子类 | 数量 | 实测结论 | 是否值得补 |
|---|---|---|---|
| **composite_implicit_autograd** | 61 | 在 PrivateUse1 key 之上被分解成已路由 leaf op;实测 conv1/2/3d、divide、true_divide、var、square、clip、selu、pad、one_hot、hstack、vstack、isfinite、kron、diag、tile、absolute、arcsinh 等**全部跑通且数值正确** | **不该**。补 dispatcher 是永不命中的死代码 |
| **no_cuda_kernel** | 12 | 无 CUDA leaf 可复用;实测 alias_copy/t_copy/diag_embed/pixel_unshuffle/select_scatter/slice_scatter/select_backward/lift_fresh_copy/equal 亦**跑通**(经 cpu_fallback 或 composite 分解) | 仅 max_pool2d_backward 受 flaggems max_pool2d **forward** 上游 bug 阻挡(与本桶无关) |
| **NOT_IN_YAML** | 9 | 该 op 名不在本 torch 版本 native_functions.yaml(别名/版本差异);bitwise_left/right_shift、copysign、new_full.Tensor、nll_loss_nd_* 实测经等价 leaf **跑通** | **不该**,无对应 schema |
| **composite_explicit_autograd** | 6 | repeat / allclose / _to_copy / copy_ / index_put / index_put_ 实测**跑通**;repeat.out 已生成 | 理论可补,但已能用,收益低 |

**核心结论**:no_dispatcher 桶几乎不是真实缺口。实测 55 个代表性 op 54 PASS，唯一未通过的 `max_pool2d_backward` 是被 flaggems `max_pool2d_with_indices` **forward** 的 stride 解析上游 bug 挡住,不属本桶职责。因此本桶**优先级应下调为最低**——补 dispatcher 收益近零。

（下方保留原始 88 个 op 全清单，供逐个查阅。）

```
__ior__.Scalar          __ior__.Tensor          __or__.Scalar
__or__.Tensor           _assert_async           _index_put_impl_
_to_copy                absolute                alias_copy
allclose                arcsinh                 arcsinh.out
arcsinh_                arctanh_                bitwise_left_shift
bitwise_right_shift     clip                    clip_
conj_physical           conv1d                  conv1d.padding
conv2d                  conv2d.padding          conv3d
conv3d.padding          copy_                   copysign
diag                    diag_embed              divide.Scalar
divide.Scalar_mode      divide.Tensor           divide.Tensor_mode
divide_.Scalar          divide_.Scalar_mode     divide_.Tensor
divide_.Tensor_mode     embedding_backward      equal
gather_backward         greater.Scalar          greater.Scalar_out
greater.Tensor          greater.out             hstack
index_put               index_put_              isclose
isfinite                kron                    lift_fresh_copy
log_sigmoid             margin_ranking_loss     max_pool2d_backward
new_full.Tensor         nll_loss_nd_backward    nll_loss_nd_forward
one_hot                 pad                     pixel_unshuffle
prelu                   quantile                relu6
repeat                  repeat_interleave.self_Tensor
repeat_interleave.self_int                      resolve_conj
resolve_neg             rms_norm                scaled_softmax_backward
scaled_softmax_forward  select_backward         select_scatter
selu                    selu_                   slice_scatter
square                  square.out              square_
t_copy                  tile                    true_divide.Scalar
true_divide.Tensor      true_divide_.Scalar     true_divide_.Tensor
var                     var.dim                 vstack
```

---

## ② type_unsupported_kwarg —— 13 个

gems 收得了参数,但某个参数类型通用 caller 表达不了。

**`Generator?`(7)** —— 随机算子,PrivateUse1 无 default generator,同 rand/randn 根因:

| op | gems qualname |
|---|---|
| `bernoulli_.float` | `bernoulli_.bernoulli_` |
| `exponential_` | `exponential_.exponential_` |
| `normal.Tensor_Tensor` | `normal.normal_tensor_tensor` |
| `normal.Tensor_float` | `normal.normal_tensor_float` |
| `normal.float_Tensor` | `normal.normal_float_tensor` |
| `normal_` | `normal.normal_` |
| `uniform_` | `uniform.uniform_` |

**`Device?` / `Layout?` / `MemoryFormat?`(5)** —— factory 元数据,但没有 shape 位置参可推断,factory caller 套用不了:

| op | gems qualname |
|---|---|
| `full_like` | `full_like.full_like` |
| `ones_like` | `ones_like.ones_like` |
| `rand_like` | `rand_like.rand_like` |
| `randn_like` | `randn_like.randn_like` |
| `zeros_like` | `zeros_like.zeros_like` |

**`Tensor?`(1)**:

| op | gems qualname |
|---|---|
| `_flash_attention_forward` | `attention.flash_attention_forward` |

---

## ③ varargs —— 12 个

gems 函数签名是 `(*args, **kwargs)`,`inspect.signature` 定不了 arity,过不了 arity 安全闸门(丢尾部参 = 静默错误)。

```
_functional_sym_constrain_range_for_size    _upsample_nearest_exact1d
asinh_          digamma_        hardswish_      i0_
log1p_          logit_          sgn_            sinh_
zero            zero.out
```

---

## ④ manual_skip —— 12 个

运行期会崩,手工排除(`FLAGGEMS_PYTHON_SKIP`)。

**device assert(8)** —— gems 内 `assert device == "cuda"`,拒 PrivateUse1:

```
maximum   minimum   _safe_softmax   upsample_linear1d
upsample_nearest1d   upsample_nearest2d   upsample_nearest3d
_upsample_bicubic2d_aa
```

**required out kwarg(1)** —— `mm.out`,gems `mm_out(a, b, *, out)` 强制 out,位置 caller 供不了。

**rng(3)** —— `rand`、`randn`、`randperm`,gems 取 `default_generators[device]` 抛 IndexError(PrivateUse1 无默认 generator);`randperm` 还 assert int dtype。

---

## ⑤ name_mismatch —— 2 个

arity-short,尾部 aten 参名对不上 gems keyword-only 参名,无法按名转发。

| op | gems qualname | aten 尾部参 | gems kwonly 参 | 原因 |
|---|---|---|---|---|
| `_grouped_mm` | `group_gemm.group_mm` | `bias`, `out_dtype` | *(无)* | gems 无 kwonly 参,无处安放 |
| `multinomial` | `multinomial.multinomial` | `generator` | `gen` | 名字不一致(且也撞 Generator? 组) |

---

## ⑥ 其余零散 —— 5 个

**foreach_tensorlist(2)** —— TensorList 类别未支持:

| op | gems qualname |
|---|---|
| `cat` | `cat.cat` |
| `stack` | `stack.stack` |

**special_optlist(1)** —— `Tensor?[]` 索引列表:

| op | gems qualname |
|---|---|
| `index.Tensor` | `index.index` |

**arity_other(2)** —— 参数重排陷阱:

| op | gems qualname | 原因 |
|---|---|---|
| `gather` | `gather.gather` | gems `out=None` 插在第 3 位,aten 第 4 参会错落进 out 槽 |
| `t_copy.out` | `t_copy.t_copy_out` | gems out 是必填位置参,`npos > with_out` |

---

## 攻关优先级参考(2026-07 实测修订)

- **~~no_dispatcher（88）~~ → 优先级最低**:实测证明这桶不是缺口——composite 分解到已路由 leaf,已能跑通且数值正确(55 探测 54 PASS)。补 dispatcher 是永不命中的死代码,**不建议投入**。仅个别 no_cuda_kernel op 若确有专用 flaggems kernel 需求,才逐个走 flaggems python 接入(非补 CUDA dispatcher)。
- **`*_like` 的 Device?/Layout?/MemoryFormat?(原 5,已接 3)** ✅ 已接 `zeros_like`/`ones_like`/`full_like`(commit 4906d22);`rand_like`/`randn_like` 无 generator 入口,排除。
- **随机 in-place(原 Generator? 组的一部分,已接 3)** ✅ 已接 `uniform_`/`exponential_`/`bernoulli_.float`(显式注入 CUDA generator)。`normal_`/`normal.*` 因 gems 硬编码 `generator=None` 不透传,排除。
- **rng factory（rand/randn/randperm/multinomial）** 无 generator 注入入口(签名无 generator 参),需给 PrivateUse1 注册 per-device generator 才能一次解决,工作量在运行时层。
- **varargs（12）** 需 gems 侧或本地维护一份显式 arity 表才能安全接入。
- **name_mismatch / arity_other / optlist / foreach（10）** 属逐个特判,收益低。
