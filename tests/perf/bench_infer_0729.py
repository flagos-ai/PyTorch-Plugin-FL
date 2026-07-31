"""
torch_fl / metax 推理 benchmark
- prefill: 固定 [bs, seq] 单次 forward 延迟 (warmup + timed)
- e2e:     model.generate 端到端 TPS (warmup + timed)

用法:
  # 原生沐曦 torch (不加载 torch_fl)
  MODE=native  python bench_infer.py
  # torch_fl (后端由 FLAGOS_BACKEND_CONFIG 决定)
  MODE=torch_fl python bench_infer.py

环境变量:
  MODEL, BS(=2), SEQ(=128), WARMUP(=3), ITERS(=10), GEN_TOKENS(=128)
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
ITERS = int(os.environ.get("ITERS", "10"))
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "128"))

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

    # native baseline (cuda:0) 在 torch2.10+MACA3.8.1 上,厂商 cat kernel 快速路径
    # 不遵守 legacy-empty 跳过规则,generate 首个 decode 步 KV cache torch.cat 崩
    # (IndexError)。torch_fl boxing/flaggems 路径靠 DropLegacyEmptyForCat 内建规避;
    # native 不走 torch_fl,故显式装形状补丁(仅改 KV placeholder 形状,不影响算子/性能,
    # 与 07-29 报告 baseline 默认开启一致)。FLAGOS_BASELINE_CAT_PATCH=0 可关。
    if MODE == "native":
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _maca_cat_patch import apply_maca_cat_patch
        if apply_maca_cat_patch():
            print("MetaX cat workaround: ON (4-D KV placeholder)")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    # 不用 device_map(会强制依赖 accelerate),直接 CPU 加载再 .to(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()
    print(f"model loaded in {time.time() - t0:.1f}s, device={next(model.parameters()).device}")

    # 固定形状输入 [BS, SEQ];在 CPU 上构造再 .to(DEVICE),避免依赖设备端
    # factory 算子(ones/randint 在 flaggems conf 里可能未注册)。
    vocab = model.config.vocab_size
    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (BS, SEQ), generator=gen).to(DEVICE)
    attn = torch.ones((BS, SEQ), dtype=torch.long).to(DEVICE)

    # ---------- 1) prefill 单次 forward 延迟 ----------
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_ids=input_ids, attention_mask=attn)
        sync()
        lat = []
        for _ in range(ITERS):
            sync()
            s = time.time()
            model(input_ids=input_ids, attention_mask=attn)
            sync()
            lat.append((time.time() - s) * 1000)
    mean = sum(lat) / len(lat)
    std = (sum((x - mean) ** 2 for x in lat) / len(lat)) ** 0.5
    print(
        f"\n[PREFILL] bs={BS} seq={SEQ} warmup={WARMUP} iters={ITERS}\n"
        f"  forward latency: mean={mean:.2f} ms  std={std:.2f} ms  "
        f"min={min(lat):.2f}  max={max(lat):.2f}"
    )

    # ---------- 2) 端到端 generate TPS ----------
    gen_kwargs = dict(max_new_tokens=GEN_TOKENS, do_sample=False, use_cache=True)
    with torch.no_grad():
        for _ in range(WARMUP):
            model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        sync()
        e2e = []
        for _ in range(ITERS):
            sync()
            s = time.time()
            out = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
            sync()
            e2e.append(time.time() - s)
    new_tok = (out.shape[1] - SEQ) * BS
    mean_s = sum(e2e) / len(e2e)
    tps = new_tok / mean_s
    print(
        f"\n[E2E generate] gen_tokens={GEN_TOKENS} bs={BS}\n"
        f"  time: mean={mean_s * 1000:.1f} ms  new_tokens/iter={new_tok}\n"
        f"  throughput: {tps:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
