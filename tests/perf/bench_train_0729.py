"""
torch_fl / metax 训练 benchmark
- 每步: forward(labels) -> loss.backward() -> optimizer.step() -> zero_grad
- 指标: 训练 TPS = bs * seq * bench_steps / 总耗时;并报 step 延迟 mean/std/min/max

用法:
  # 原生沐曦 torch (不加载 torch_fl)
  MODE=native   python bench_train.py
  # torch_fl (后端由 FLAGOS_BACKEND_CONFIG 决定)
  MODE=torch_fl python bench_train.py

环境变量:
  MODEL, BS(=2), SEQ(=128), WARMUP(=3), STEPS(=10), LR(=1e-5)
"""

import os
import time

# MetaX 上必须先 import torch_fl 再 import torch(预加载 cudart 符号 shim,
# 解决 fork libtorch CUDA12 与 cu-bridge CUDA11.6 的 ABI 不兼容)。
MODE = os.environ.get("MODE", "native")
if MODE == "torch_fl":
    import torch_fl  # noqa: F401

import torch

# FlagGems 的 triton autotune 用 torch.cuda.Event 计时,但输入是 flagos
# (PrivateUse1) tensor 时 event.record 会失败("Unknown device: 100" /
# "context is destroyed")。这里把 do_bench 换成 wall-clock + synchronize 版,
# 让 autotune 仍能测时选 config,同时避开 CUDA event。仅在 flaggems 路径需要。
if MODE == "torch_fl" and os.environ.get("FLAGOS_USE_FLAGGEMS") == "1":
    def _do_bench_wall(fn, *args, quantiles=None, return_mode="mean", **kwargs):
        fn()
        torch.cuda.synchronize()
        n = 5
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000 / n
        if quantiles is not None:
            return [ms] * len(quantiles)
        return ms

    # do_bench 在多个模块里被 `from ..testing import do_bench` 绑定成局部名字,
    # 逐个替换。autotuner 是 flaggems libtuner 实际走到的那个。
    import triton.testing as _tt
    import triton.runtime.autotuner as _ta

    _tt.do_bench = _do_bench_wall
    _ta.do_bench = _do_bench_wall

MODEL = os.environ.get("MODEL", "/data/nfs/Qwen3-0.6B")
BS = int(os.environ.get("BS", "2"))
SEQ = int(os.environ.get("SEQ", "128"))
WARMUP = int(os.environ.get("WARMUP", "3"))
STEPS = int(os.environ.get("STEPS", "10"))
LR = float(os.environ.get("LR", "1e-5"))

if MODE == "torch_fl":
    DEVICE = "flagos:0"

    def sync():
        torch_fl.flagos.synchronize()

    print(
        f"[torch_fl] FlagGems enabled={torch_fl.is_flaggems_enabled()}  "
        f"registered ops={len(torch_fl.get_registered_ops())}  "
        f"config={os.environ.get('FLAGOS_BACKEND_CONFIG', '(auto)')}"
    )
else:
    # 沐曦原生 torch: fork 版把设备暴露成 cuda
    DEVICE = "cuda:0"

    def sync():
        torch.cuda.synchronize()

    print(f"[native] torch={torch.__version__} device={DEVICE}")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 显式启用 swap 路径:flagos 设备上 .to() 走 swap 能成功;传统 copy 路径会
    # 触发 "detach: backend not registered"。默认值不保证为 True,显式设置。
    torch.__future__.set_swap_module_params_on_conversion(True)

    t0 = time.time()
    _ = AutoTokenizer.from_pretrained(MODEL)
    # fp32 训练;不用 device_map(会强制依赖 accelerate),CPU 加载再 .to(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model = model.to(DEVICE)
    model.train()
    print(
        f"model loaded in {time.time() - t0:.1f}s, "
        f"device={next(model.parameters()).device}"
    )

    # 固定形状输入 [BS, SEQ];在 CPU 上构造再 .to(DEVICE),避免依赖设备端
    # factory 算子(ones/randint 在 flaggems conf 里可能未注册)。
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (BS, SEQ), generator=g).to(DEVICE)
    attn = torch.ones((BS, SEQ), dtype=torch.long).to(DEVICE)
    labels = input_ids.clone()

    # 冻结不参与反向的参数(与 test_qwen3_train.py 一致),避免 AdamW 更新
    # 无 grad 参数时报错。用一次 forward/backward 探测。
    with torch.enable_grad():
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                    use_cache=False)
        out.loss.backward()
    frozen = 0
    for p in model.parameters():
        if p.grad is None:
            p.requires_grad = False
            frozen += 1
        else:
            p.grad = None
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"frozen {frozen} params; {trainable:.1f}M / {total:.1f}M trainable")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    def train_step():
        outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                        use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    # ---------- warmup ----------
    for _ in range(WARMUP):
        train_step()
    sync()

    # ---------- timed steps ----------
    step_t, losses = [], []
    for _ in range(STEPS):
        sync()
        s = time.time()
        loss = train_step()
        sync()
        step_t.append((time.time() - s) * 1000)
        losses.append(loss.item())

    mean = sum(step_t) / len(step_t)
    std = (sum((x - mean) ** 2 for x in step_t) / len(step_t)) ** 0.5
    total_tok = BS * SEQ * STEPS
    tps = total_tok / (sum(step_t) / 1000)
    print(
        f"\n[TRAIN] bs={BS} seq={SEQ} fp32 warmup={WARMUP} steps={STEPS}\n"
        f"  step latency: mean={mean:.2f} ms  std={std:.2f} ms  "
        f"min={min(step_t):.2f}  max={max(step_t):.2f}\n"
        f"  loss: first={losses[0]:.4f} last={losses[-1]:.4f}\n"
        f"  throughput: {tps:.2f} tok/s  (tokens={total_tok})"
    )


if __name__ == "__main__":
    main()
