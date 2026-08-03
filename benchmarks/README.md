# Benchmarks

## FlagGems dispatch-overhead microbenchmark

Compares the **host-side dispatch cost** of three paths for the same op on the
same device (`flagos:0`, shared GPU memory):

- **cuda** — default vendor conf (`backends_cuda.conf`), pure C++ dispatch into
  the vendor `libtorch_cuda` kernel.
- **gems_py** — `FLAGOS_USE_FLAGGEMS=1` (`backends_flaggems.conf`), Python
  dispatch: `kFlagOsPython` crosses into CPython + pybind to launch the FlagGems
  Triton kernel.
- **gems_cpp** — `FLAGOS_USE_FLAGGEMS_CPP=1` (`backends_flaggems_cpp.conf`), C++
  dispatch: `kFlagOs` boxes flagos→cuda metadata and calls the FlagGems C++
  runtime (`liboperators.so`, TritonJIT, **no GIL/pybind**). Only the 18 Stage-A
  ops route here; the rest fall back to `gems_py`.

```bash
python benchmarks/flaggems_dispatch_bench.py            # defaults: submit=300 e2e=100

# gems_cpp column additionally needs a FLAGGEMS_KERNEL=ON wheel + (one line):
FLAGGEMS_SOURCE_DIR=/path/to/FlagGems-src/src/flag_gems \
LD_LIBRARY_PATH=/path/to/FlagGems-src/cpp/build/lib:$LD_LIBRARY_PATH \
python benchmarks/flaggems_dispatch_bench.py --submit 500 --e2e 200
```

The driver runs `flaggems_dispatch_overhead.py` once per backend in its own
subprocess (routing is fixed at import). If the C++ runtime env is missing it
skips `gems_cpp` instead of failing.

### Metrics

- **submit** — issue N ops back-to-back, synchronize once at the end. The async
  queue hides kernel time, so this isolates host dispatch cost: GIL + pybind +
  a CPython frame (gems_py) vs a thin C++ dispatcher (cuda / gems_cpp).
- **e2e** — synchronize after every op: full per-op wall time including kernel.
- **py_tax** = `gems_py − cuda` — the Python dispatch overhead vs vendor.
- **cpp_save** = `gems_py − gems_cpp` — how much of that tax the C++ dispatch
  actually recovers (only meaningful on ops marked `*`).

Inputs are built on CPU and moved to `flagos:0` to sidestep the RNG shim (a
CPU-torch build cannot create a CUDA generator directly).

### Measured (NVIDIA A100, torch 2.10.0+cpu + flagos PrivateUse1, µs/op, submit=500)

`*` = op routes to the C++ `kFlagOs` backend under `backends_flaggems_cpp.conf`.

| op | cat | cuda | gems_py | gems_cpp | py_tax | cpp_save | gpy_e2e | gcpp_e2e |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| add[512²] | elementwise | 8 | 92 | 95 | 84 | — | 103 | 107 |
| add[4096²] | elementwise | 149 | 150 | 150 | 1 | — | 240 | 244 |
| mul[512²] | elementwise | 8 | 17 | 19 | 9 | — | 25 | 27 |
| abs[512²] | elementwise | 10 | 80 | 83 | 70 | — | 86 | 88 |
| neg[512²] | elementwise | 8 | 78 | 81 | 70 | — | 84 | 87 |
| sum.dim[1024²] `*` | reduction | 10 | 49 | **11** | 38 | **38** | 55 | **21** |
| softmax[1024²] `*` | reduction | 8 | 40 | **11** | 32 | **30** | 49 | **21** |
| mm[512²] `*` | matmul | 27 | 65 | 110 | 38 | **−45** | 123 | 128 |
| mm[4096²] `*` | matmul | 7219 | 8059 | 8176 | 840 | **−117** | 8140 | 8201 |

(unmarked ops still use the Python path under the cpp conf, so their tiny
cpp_save is measurement noise — shown as `—`.)

### Reading the numbers

- **Python dispatch tax is real and roughly fixed: ~70–90 µs/op.** On small
  elementwise ops vendor submits in ~8 µs; the Python path needs ~80–100 µs —
  a ~10× host gap.
- **C++ dispatch delivers big on host-bound reductions.** `sum.dim` and
  `softmax` drop from ~49/40 µs to ~11 µs — `gems_cpp` submit essentially
  matches vendor (10 µs), recovering ~30–38 µs (the whole py_tax). End-to-end
  they more than halve (55→21, 49→21) because the C++ path also skips the
  Python-side launch bookkeeping the per-op sync would otherwise expose.
- **C++ dispatch is a net loss on `mm`** (`cpp_save` **negative**). The C++
  TritonJIT `mm` path here is *slower to submit* than the Python one (110 vs
  65 µs at 512²; 8176 vs 8059 µs at 4096²). The matmul kernel dominates either
  way, so the host difference is noise relative to a 7 ms kernel — but it shows
  the C++ dispatch is not automatically cheaper: for compute-bound ops whose
  kernel already hides the host cost, routing them to `kFlagOs` buys nothing and
  can cost a little (extra boxing + a C++ TritonJIT launch that isn't autotuned
  the way the Python FlagGems path is).
- **e2e reflects kernel quality, not just dispatch.** FlagGems Triton kernels
  are not faster than cuBLAS/vendor for these shapes, so both gems columns trail
  `cuda` end-to-end; C++ dispatch removes the host tax, not a kernel-speed gap.

**Takeaway:** the win is exactly where Stage A aimed once you look per-category —
**host-bound reductions/elementwise ops in eager per-op loops** (sum/softmax:
~3–4× lower submit, ~2.5× lower e2e). For matmul and large reductions the async
pipeline already hides the Python cost, so `kFlagOs` should *not* be enabled for
them — which is why `backends_flaggems_cpp.conf` is a hand-picked mixed conf, not
a blanket switch. The negative `mm` result is a concrete argument for keeping the
per-op routing table selective (Stage C) rather than C++-dispatching everything.
