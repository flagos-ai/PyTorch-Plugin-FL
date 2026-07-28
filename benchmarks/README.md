# Benchmarks

## FlagGems dispatch-overhead microbenchmark

Quantifies the **host-side cost** of the Python FlagGems path (`kFlagOsPython`,
`FLAGOS_USE_FLAGGEMS=1`) versus the vendor path (`kCuda`, pure C++ dispatch), to
estimate how much a hypothetical C++ FlagGems dispatch (`kFlagOs`) could recover.

```bash
python benchmarks/flaggems_dispatch_bench.py            # defaults: submit=300 e2e=100
python benchmarks/flaggems_dispatch_bench.py --submit 500 --e2e 150
```

The driver runs `flaggems_dispatch_overhead.py` once per backend in its own
subprocess (routing is fixed at import) and prints a side-by-side table.

### Metrics

- **submit** — issue N ops back-to-back, synchronize once at the end. The async
  queue hides kernel time, so this isolates host dispatch cost: GIL acquire +
  pybind tensor/arg marshalling + a CPython frame (Python path) vs a thin C++
  dispatcher (cuda path).
- **e2e** — synchronize after every op: full per-op wall time including kernel.
- **host_delta** = `gems_submit − cuda_submit`. This is the realistic ceiling on
  what a C++ FlagGems dispatch could save, since both paths then launch the same
  class of GPU work.

Inputs are built on CPU and moved to `flagos:0` to sidestep the RNG shim (a
CPU-torch build cannot create a CUDA generator directly).

### Measured (NVIDIA A100, torch 2.10.0+cpu + flagos PrivateUse1, µs/op)

| op | cat | cuda_sub | gems_sub | host_delta | cuda_e2e | gems_e2e |
|---|---|---:|---:|---:|---:|---:|
| add[512×512] | elementwise | 8 | 96 | **88** | 17 | 101 |
| add[4096×4096] | elementwise | 149 | 149 | **0** | 162 | 242 |
| mul[512×512] | elementwise | 8 | 91 | **83** | 17 | 96 |
| abs[512×512] | elementwise | 9 | 80 | **71** | 18 | 87 |
| neg[512×512] | elementwise | 8 | 80 | **72** | 17 | 87 |
| sum.dim[1024×1024] | reduction | 10 | 51 | **41** | 23 | 56 |
| softmax[1024×1024] | reduction | 11 | 39 | **29** | 24 | 48 |
| mm[512×512] | matmul | 35 | 68 | **32** | 54 | 120 |
| mm[4096×4096] | matmul | 7220 | 8063 | **843** | 7320 | 8155 |

### Reading the numbers

- **Python dispatch tax is ~70–90 µs/op, roughly fixed.** On small elementwise
  ops the vendor path submits in ~8 µs; the Python path needs ~80–100 µs. That
  ~10× host gap is exactly what C++ dispatch would erase.
- **The tax only matters when the kernel is short.** `add[4096×4096]` has a
  ~150 µs kernel that fully hides the host cost (`host_delta ≈ 0` in the async
  submit column) — but it still shows up end-to-end (242 vs 162) because per-op
  sync stops the pipeline from hiding it.
- **Compute-bound ops see almost nothing.** `mm[4096×4096]`: 843 µs delta on a
  7.2 ms kernel is ~11% of submit and <1% relative once the pipeline runs.
- **e2e also reflects kernel quality, not just dispatch.** flaggems Triton
  kernels here are not faster than cuBLAS/vendor for these shapes, so gems_e2e
  > cuda_e2e; C++ dispatch removes the host tax but not a kernel-speed deficit.

**Takeaway:** C++ FlagGems dispatch is worth it precisely for **host-bound
small/elementwise/small-reduction ops** in eager, per-op hot loops (and more so
under GIL contention: multi-stream, DataLoader workers). For matmul and large
reductions the async pipeline already hides the Python cost, so C++ dispatch
buys little. This matches the estimate that ~150–200 of the 319 flagos_python
ops (the small/host-bound ones) are the ones actually worth C++-dispatching.
