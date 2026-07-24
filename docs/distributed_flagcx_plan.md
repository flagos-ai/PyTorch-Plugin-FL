# FlagCX 接入 + NCCL 兜底：多硬件统一分布式通信设计

本文件描述 `torch_fl.distributed` 在 flagos（PrivateUse1）设备上如何统一接入
FlagCX，并在 FlagCX 不可用时回退到各硬件厂商（vendor）的原生通信后端。目标是
让 nvidia / metax / ascend 三类硬件共用同一套上层 API，同时把通信正确性缺口补齐。

> 状态：设计 + 第一阶段重构（纯 Python，不依赖 flagcx 环境即可验证正确性）。
> FlagCX 原生注册 / ascend view 的实机验证留待有 flagcx 与多卡环境时进行。

---

## 1. 背景与现状

### 1.1 零拷贝桥接

flagos 张量与 CUDA 张量共享同一块显存。`torch_fl/csrc/module.cc` 的
`flagos_to_cuda_view_impl` 用同一个 `data_ptr` 构造一个带 `DispatchKey::CUDA`
的 tensor（保留对原 flagos tensor 的引用防止显存被释放）。因此通信后端拿到的
其实是 CUDA tensor，完全感知不到 flagos 的存在。

### 1.2 原有通信路径（重构前）

`torch_fl/distributed.py` 的 `init_process_group` 走两条腿：

- `backend="nccl"`：正常 `dist.init_process_group("nccl")`，然后
  `_register_privateuseone_backend` 把 cuda 上的 backend 复制注册到
  privateuseone 设备，再 `_patch_dist_collectives` 把 5 个集合通信 API
  monkeypatch 成「先转 cuda view 再调原函数」。
- `backend="flagcx"`：`import flagcx` 触发 entry-point 注册，用
  `backend="cpu:gloo,cuda:flagcx"` 初始化。

`_patch_dist_collectives` 只覆盖了 5 个 API：`all_reduce`、`broadcast`、
`reduce`、`all_gather_into_tensor`、`reduce_scatter_tensor`。

### 1.3 vendor 探测的权威来源

`torch_fl/__init__.py` 的 `_patch_flaggems_codegen_config()` 在 import 期就把
`GEMS_VENDOR` 环境变量设为 `nvidia` / `metax` / `ascend` 之一。分布式层应
**直接复用** 这个变量，不要另起一套硬件探测。

---

## 2. 已识别的问题

1. **BackendType 硬编码 NCCL**：`_register_privateuseone_backend` 写死
   `BackendType.NCCL`，在 ascend（HCCL 非 NCCL 强类型）上不成立。
2. **view 目标写死 cuda**：`_ensure_cuda` 无脑 flagos→cuda。nvidia/metax
   可行（metax 走 maca 的 libtorch_cuda 兼容层），但 **ascend 没有 CUDA
   兼容层**，这条路根本不通。
3. **API 覆盖面缺口**：未被 patch 的集合通信 API 一旦被调用，会拿到
   privateuseone tensor 直接崩溃。缺口包括 `all_gather`（list 版）、
   `gather`/`scatter`、`all_to_all[_single]`、`send`/`recv`/`isend`/`irecv`、
   `barrier`，以及最隐蔽的 `torch.ops._c10d_functional.*`（torch.compile /
   DTensor / FSDP2 编译路径都走这条）。
4. **回退逻辑缺失**：flagcx 不可用时没有自动降级到 vendor 原生后端。

---

## 3. 目标架构

```
                flagos_dist.init_process_group(backend="auto")
                                │
                ┌───────────────┴───────────────┐
                │   _resolve_backend()           │  读 GEMS_VENDOR + 用户请求
                │   FlagCX 优先，vendor 原生兜底  │
                └───────────────┬───────────────┘
     ┌──────────────────┬───────┴────────┬──────────────────┐
  flagcx 可用?        nvidia            metax             ascend
     │ 是               │ nccl 兜底       │ mccl 兜底        │ hccl 兜底
统一 flagcx backend      │ (NCCL 类型)     │ (走 maca         │ (CUSTOM 类型,
(CUSTOM 类型)            │                │  libtorch_cuda)  │  torch_npu)
     │                   └────────┬───────┘                  │
     │              flagos→cuda view (共享 data_ptr)    flagos→npu view
     │                            │                    (无 CUDA 兼容层!)
     └──────────── 原生认 privateuseone，免 view ──────────────┘
```

---

## 4. 三个改造点

### 4.1 `_resolve_backend()`：后端解析 + 回退（兜底语义 A）

收敛所有「选谁 / 降级」逻辑到一个纯 Python 函数：

- 读 `GEMS_VENDOR` 得到 vendor，映射到 vendor 原生后端：
  - `nvidia` → `nccl`
  - `metax`  → `nccl`（maca 的 mccl 在 PyTorch 眼里就是 NCCL backend，底层链
    的是 mccl）
  - `ascend` → `hccl`
- 用户显式指定 `nccl`/`hccl`/`flagcx` 时尊重用户。
- `auto`（默认）优先 flagcx。
- flagcx 请求路径 `import flagcx` 失败时，warning 并回退到 vendor 原生后端。

返回 `(实际后端字符串, vendor)`。

### 4.2 `_register_privateuseone_backend()`：BackendType 按 vendor 分流

- BackendType 映射：`flagcx`→`CUSTOM`、`nccl`→`NCCL`、`hccl`→`CUSTOM`。
- 探测原 backend 的设备：ascend 从 `privateuseone` 取，其余从 `cuda` 取。

### 4.3 `_patch_dist_collectives()`：view 分流 + API 覆盖面补全（兜底语义 B）

**(a) view 目标按 vendor 分流**：抽象成 `_ensure_comm_tensor(t, vendor)`：

- `ascend` → `_flagos_to_npu_view(t)`（需新增 C++ 实现，或走 flagcx 原生注册
  彻底免 view）
- 其余 → `_flagos_to_cuda_view(t)`（复用现有）

**(b) API 覆盖面补全**：用通用 patch 生成器，遍历一张
`(函数名, 哪些位置/关键字参数是 tensor)` 表批量包裹，避免每个函数手写。至少覆盖：

- `all_reduce`、`broadcast`、`reduce`、`all_gather_into_tensor`、
  `reduce_scatter_tensor`（原有 5 个）
- `all_gather`（list 版）、`gather`、`scatter`
- `all_to_all`、`all_to_all_single`
- `send`、`recv`、`isend`、`irecv`
- `barrier`（device_ids 参数）
- `torch.ops._c10d_functional.*`（functional collectives）

---

## 5. FlagCX 原生注册（免 view）——优先验证的假设

若 flagcx 的 adaptor 只认 `data_ptr + stream`、不校验 device type，则可给
privateuseone 直接注册 flagcx backend，**patch 与 view 全都不需要**，且天然覆盖
所有集合通信 API。这对 ascend 尤其有价值（省掉 `_flagos_to_npu_view` 的 C++
工作）。拿到 flagcx 环境后，这是第一个该验证的点。

---

## 6. 各 vendor 落地状态

| vendor | flagcx 路径 | 原生兜底 | view 转换 | 主要缺口 |
|--------|------------|---------|-----------|---------|
| nvidia | 现成可跑 | nccl（现成） | flagos→cuda（现成） | 仅 API 覆盖面 |
| metax  | 应可复用 | "nccl"@maca | flagos→cuda（走 maca） | 需实测 mccl |
| ascend | **建议主路径** | hccl（CUSTOM） | **缺 flagos→npu** | view 或原生注册二选一 |

---

## 7. 实施顺序

1. **重构 `distributed.py`**（不改 nvidia 行为）：抽出 `_resolve_backend` /
   vendor 分流的 `_register_privateuseone_backend` / `_ensure_comm_tensor`，
   搭好架构骨架。
2. **补全 collective API 覆盖面**（含 functional collectives）：纯 Python、
   不依赖硬件，当前环境即可验证正确性。
3. **flagcx 环境验证**：验证原生注册免 view 假设 → 决定 ascend 走原生注册
   还是补 `_flagos_to_npu_view`。
4. **metax / ascend 实机验证**兜底路径。

---

## 8. 公共 API（重构后）

```python
import torch_fl.distributed as flagos_dist

# backend 取值：
#   "auto"   -> flagcx 优先，失败回退 vendor 原生（推荐）
#   "flagcx" -> 强制 flagcx，失败回退 vendor 原生
#   "nccl"   -> 强制 nccl（nvidia/metax）
#   "hccl"   -> 强制 hccl（ascend）
flagos_dist.init_process_group(backend="auto")

model = flagos_dist.DistributedDataParallel(model)
flagos_dist.move_buffers_to_device(model, "flagos:0")
```
