# FlagGems 未接入算子清单

FlagGems `_FULL_CONFIG` 共 **433** 个算子,当前 **301 已路由**到 `flagos_python` 路径,**132 个对接不上**。本文按原因分桶列出这 132 个,供后续逐批攻关。

数据由 `discover_flaggems_ops()`(`scripts/codegen_ops.py`)的逐分支拒绝逻辑对账得出,分桶与实际 codegen 拒绝完全一致。

| 桶 | 数量 | 一句话原因 |
|---|---|---|
| ① no_dispatcher | 88 | aten 侧没生成 dispatcher(该 schema 未进 `backends_cuda.conf`) |
| ② type_unsupported_kwarg | 13 | 参数类型通用 caller 表达不了(Generator?/Device?/Layout?/MemoryFormat?/Tensor?) |
| ③ varargs | 12 | gems 签名 `(*args, **kwargs)`,arity 无法内省 |
| ④ manual_skip | 12 | 运行期崩溃,手工排除(device assert / 必填 out / rng) |
| ⑤ name_mismatch | 2 | 尾部 aten 参名对不上 gems keyword-only 参名 |
| ⑥ 其余零散 | 5 | foreach / optlist / arity 重排陷阱 |
| **合计** | **132** | |

---

## ① no_dispatcher —— 88 个

gems 有实现,但 aten 侧 codegen **没生成 dispatcher**。要先在 CUDA codegen 里补出该 op 的 dispatcher,才谈得上路由到 flaggems。这是最大头。

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

## 攻关优先级参考

- **no_dispatcher（88）** 是最大且最独立的一块:补齐 CUDA 侧 dispatcher 后可批量解锁,但工作量在 aten codegen 侧,非 flaggems 转发层。
- **Generator? / rng（7 + 3 = 10）** 同根:需要给 PrivateUse1 注册 per-device generator,一次解决随机算子组。
- **`*_like` 的 Device?/Layout?/MemoryFormat?（5）** 可仿 factory caller 扩展(从输入 tensor 推 shape + 注入 device=flagos）。
- **varargs（12）** 需 gems 侧或本地维护一份显式 arity 表才能安全接入。
- **name_mismatch / arity_other / optlist / foreach（10）** 属逐个特判,收益低。
