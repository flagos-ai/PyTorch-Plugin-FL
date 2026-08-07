# D-Robotics RDK BPU (Horizon BPU) backend

The BPU is the one platform in this repo where acceleration does **not** come
from operator kernels. Its BPU executes whole compiled graphs, so `torch_fl`
provides a real device (UCP-backed memory, device/stream layer) plus a
`torch.compile` backend, and every eager op runs on the CPU.

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
ACCELERATOR=bpu pip install --no-build-isolation -e .
```

```python
import torch, torch_fl

model = MyNet().eval()
compiled = torch.compile(model, backend="bpu")
out = compiled(torch.randn(1, 3, 224, 224))
```

## torch version

**Pinned to the 2.10 series** (`TORCH_PIN = "torch>=2.10,<2.11"` in `setup.py`,
mirrored in `pyproject.toml`'s build requires). This is the same pin every
platform in this repo carries, and it is enforced, not advisory.

The pin exists because the checked-in `csrc/aten/generated/*` bindings are
generated against one specific ATen surface. A newer torch drifts from them and
fails as a wall of compile errors at build time rather than a clean resolver
error, so the pin is what turns a confusing build break into an install-time
message. Moving to a newer torch is a deliberate act: re-run
`scripts/codegen_ops.py`, do not hand-edit the generated files.

The board runs `torch 2.10.0+cpu` on `/home/sunrise/miniconda3/bin/python3.14`
(the cp314 aarch64 wheel exists on PyPI).

## Why not per-op kernels

`hbDNNInferV2` takes a model handle and a tensor array — a compiled `.hbm`
artifact, not a Conv2d argument list. There is no entry point that computes one
convolution, so there is nothing for a `PrivateUse1` kernel to call. This
mirrors `tsingmicro`: `csrc/aten/register.cc` skips the generated
`register.inc` under `USE_BPU`, and every aten op reaches `cpu_fallback`.

Convolution is the one exception, and not by choice. `aten::convolution`
dispatches `PrivateUse1` to `convolution_overrideable`, whose only other kernel
is a `CompositeExplicitAutograd` stub that raises `NotImplementedError`. The
boxed fallback cannot rescue it — moving the arguments to CPU and redispatching
the *same* op lands back on the stub — so `register.cc` registers two small
wrappers that cross to CPU and call `at::convolution` instead. Without them
`conv2d` on a flagos tensor raises rather than falling back.

## Compile pipeline

```
Dynamo -> AOTAutograd -> aten FX graph
       -> partition           (torch_fl/accelerator/bpu/partition.py)
       -> freeze weights      (params/buffers become ONNX initializers)
       -> ONNX export         (compiler.py, decompose.py)
       -> int8 Q/DQ insertion (qdq.py, calibrate.py)
       -> hbdk4               -> .hbm, cached by graph structure
       -> hbm_runtime         (runtime.py)
```

Two parts of this are load-bearing rather than optimizations:

**Quantization is a precondition.** hbdk4's `convert(advice=True)` says it
outright: `"lower to cpu. P.S. The type of hbir.conv's fin is f32, which should
be si8, si16 on bpu."` A float artifact compiles fine and then runs conv on the
CPU, so the BPU sits idle. Q/DQ insertion is what moves the MAC work onto the
device — measured 6.1 ms to 0.64 ms on a 2-conv net. Set
`FLAGOS_BPU_QUANTIZE=0` for bit-exact float artifacts with no speedup.

**Weight freezing is what makes offload a win at all.** AOTAutograd lifts every
parameter and buffer to a graph input, so a 2-conv block crosses the partition
boundary with 13 tensors instead of 1 — copied on every call, and emitted as
ONNX graph *inputs* rather than initializers, which also blocks the int8 fold.
Before freezing, BPU offload measured 3x *slower* than eager (25.7 ms vs
2.8 ms); after, 0.849 ms.

Partitions that cannot be compiled stay in the graph and run eagerly, so a
missing or failing hbdk4 costs performance and never correctness. Pass
`strict=True` to `bpu_backend` to raise instead.

## Compiling on the board

hbdk4 ships **x86_64-only** wheels (`hbdk4_compiler-4.7.5-cp310/cp311-manylinux_2_17_x86_64.whl`);
the aarch64 wheels are runtime-only. So compiling on the board means running an
x86_64 Python under an emulator — but it does work, on the **stock kernel**, with
no VM and no cross-compile host.

One command sets it up:

```bash
scripts/setup_bpu_hbdk4.sh --wheels /path/to/oe/wheels
export FLAGOS_BPU_X86_PYTHON=~/hbdk4-x86/python/bin/python3.11
export FLAGOS_BPU_X86_EMULATOR=~/hbdk4-x86/bin/box64
```

Verify with:

```bash
python -c "from torch_fl.accelerator.bpu.compiler import find_hbdk; print(find_hbdk())"
# -> x86-emul
```

The wheels come from the D-Robotics OE package (`oe-package-*.tgz` on
`ftp://oeftp@sdk.d-robotics.cc/`); the script needs `hbdk4_compiler-*cp311*` and
`hbdk4_march-*`.

### Why the setup is not just "apt install box64"

Four things have to line up. Each one fails in a way that looks unrelated to the
real cause, so they are worth naming.

**1. box64 must be built from source; the packaged one is too old.** Debian and
Ubuntu ship 0.2.6 (Jan 2024), which had its page size fixed at build time and
aborts immediately:

```
Error: PageSize configuration is wrong: configured with 4096, but got 65536
```

Current box64 reads the host page size at runtime (`box64_pagesize =
sysconf(_SC_PAGESIZE)`) and maps 4 KB-aligned x86 `PT_LOAD` segments onto 64 KB
pages itself. Verified with **v0.4.5**: the same x86_64 binary that 0.2.6 refuses
runs correctly. **No kernel rebuild is needed** — earlier revisions of this doc
described one, and it is unnecessary.

qemu-user has no equivalent fix. It still fails with `SIGBUS` on any `.so` with
4 KB-aligned segments, and `pip` segfaults under it. box64 is the working path.

**2. `libhbtl.so` needs an explicit `RTLD_GLOBAL` preload.** `_hbdk*.so` expects
`hbtl::Storage::createExternal` and `hbtl::getStrides` to be resolvable, but does
not list `libhbtl.so` in its own `DT_NEEDED` — it inherits them transitively
through `libHBDKPythonCAPI.so`, which box64 does not reproduce. Without the
preload:

```
Error: Symbol _ZN4hbtl7Storage14createExternalE... not found,
       cannot apply R_X86_64_JUMP_SLOT
ImportError: Cannot dlopen(".../_hbdk.cpython-311-x86_64-linux-gnu.so")
```

`compiler.py` handles this: `x86_env()` puts the `_mlir_libs` directory on
`BOX64_LD_LIBRARY_PATH` (box64 resolves guest libraries through its own search
path, so `RUNPATH=$ORIGIN` is not enough) and the compile driver does the
`ctypes.CDLL(..., RTLD_GLOBAL)` preload.

**3. numba must be *absent* and stubbed, not installed.** hbdk4's ONNX entry
point imports numba unconditionally, but real numba imports `llvmlite.binding`,
whose x86 LLVM JIT segfaults under box64. The crash is in llvmlite's JIT setup —
not in numba, and not in hbdk4.

Stubbing it out is safe because hbdk4 only ever *calls* numba for custom/numba
ops: `compile_numba()` returns the module untouched when `has_numba_op()` is
false, which is always true for a graph exported from ONNX standard operators.
The stubs raise if actually invoked, so a graph that genuinely needs numba fails
loudly rather than miscompiling.

**4. torch is stubbed for the same reason.** hbdk4's tracing module imports torch
at module scope but only uses it for `isinstance` checks and `torch.jit.trace` in
the custom-op path. A real x86_64 torch would work but costs several hundred MB
in the emulated environment for code that never runs. Note the stub is the
*emulated* interpreter's torch, entirely separate from the board's aarch64 torch
that torch_fl itself runs on.

The stubs live in `~/hbdk4-x86/stubs` and are appended to `PYTHONPATH`, never
prepended, so a real numba or torch installed in the guest would win.

### One tolerated failure

hbdk4's `compile()` ends by loading the artifact back through hbrt4 to validate
it, which claims BPU device memory. Under emulation that can fail:

```
hbrt4_py.Hbrt4PyError: ... Cannot malloc bpu memory with length 52496 bytes:
                           AllocError { len: 135168 }
```

This happens *after* a complete `.hbm` has been written. The compile driver
tolerates exactly this case — non-empty output file plus an exception from the
validation step — because the artifact is the deliverable and torch_fl then loads
it with the board's native aarch64 runtime, which is a stricter check.

### Caching

`find_hbdk()` tries a native import, then a CLI driver, then the emulator
(`FLAGOS_BPU_X86_EMULATOR` first, then `box64`, `qemu-x86_64-static`,
`qemu-x86_64`), caching the result. When none is reachable the backend logs a
warning and leaves every partition on the CPU.

Emulated compilation is slow, which is why `compile_partition` caches artifacts
under `~/.cache/torch_fl_bpu` keyed by graph structure, input signature, march,
and the activation-scale set. Float and int8 builds never share a cache entry.

## Calibration

Without calibration, activations use `FLAGOS_BPU_ACT_SCALE` (default 0.05,
covering roughly ±6.35 — wide enough for post-BN/ReLU activations). For better
accuracy, measure real ranges:

```python
from torch_fl.accelerator.bpu.calibrate import calibrate_onnx
scales = calibrate_onnx("partition.onnx", samples=[x1, x2, ...])
```

Scales key on **ONNX** tensor names, which have no stable relation to torch
module names, so calibration runs on the exported graph via onnxruntime rather
than on the eager module. `calibrate_module` collects per-module ranges when the
torch-side view is what you want.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLAGOS_BPU_X86_PYTHON` | unset | x86_64 python with `hbdk4-compiler`, run under box64 |
| `FLAGOS_BPU_X86_EMULATOR` | unset | path to a box64 binary; needed because the distro 0.2.6 is too old and a self-built one is not on `PATH` |
| `FLAGOS_BPU_X86_STUBS` | `<x86 python>/../../stubs` | numba/torch import stubs for the emulated interpreter |
| `FLAGOS_BPU_MARCH` | `nash-p` | BPU micro-architecture (`nash-p`=BPU, `nash-e`=S100, `nash-m`=S100P) |
| `FLAGOS_BPU_QUANTIZE` | `1` | int8 Q/DQ insertion; `0` compiles float (bit-exact, no BPU speedup) |
| `FLAGOS_BPU_ACT_SCALE` | `0.05` | fallback activation scale for uncalibrated tensors |
| `FLAGOS_BPU_CACHE` | `~/.cache/torch_fl_bpu` | `.hbm` artifact cache |

## Runtime notes

- **Device memory is host-mapped.** `hbUCPMallocCached` returns an
  `hbUCPSysMem` carrying both a host-writable `virAddr` and a `phyAddr`, and it
  works unprivileged. So `Memcpy` is a plain `memcpy` plus cache maintenance
  (invalidate before reading what the BPU wrote, clean after writing what it
  will read) — no bounce buffer, and a zero-copy inference path is possible
  because a tensor's storage can *be* the BPU's memory.
  `FlagosBPUPhysicalAddress()` exposes the virtual-to-physical mapping for
  building an `hbDNNTensor` without a copy.
- **Four BPU cores, one device.** `hb_bpu_core_open()` takes a core *mask* with
  a scheduling policy, and UCP memory is SoC-wide, so cores are a scheduling
  detail rather than separate devices. `GetDeviceCount` returns 1, and 0 when
  the driver is absent (`hb_bpu_core_num() == 0`).
- **Submission is synchronous** (`hbUCPSubmitTask` + `WaitTaskDone`), so there
  is no async copy engine: `MemcpyAsync` is a synchronous copy, streams are a
  single sentinel handle, and event timestamps are real `steady_clock` readings
  — which makes `EventElapsedTime` genuinely meaningful here.
- **Cached blocks leak at process exit, deliberately.** The caching allocator is
  a function-local static, so its destructor runs from `__run_exit_handlers`, by
  which point `libhbucp`'s own `FINI_ARRAY` teardown may have released the heap
  the UCP blocks live in — `hbUCPFree` then aborts with `double free or
  corruption (fasttop)`. `caching_device_allocator.cc` skips the release under
  `USE_BPU` (as it already does for tsingmicro). Calling
  `torch_fl._C._empty_cache()` before exit frees the same blocks cleanly, which
  is what confirms only the exit ordering is at fault. The kernel reclaims the
  carveout when the fd closes.

## Measured performance

A 6-layer conv stack at 224x224, quantized with frozen weights, compiled
**on the board** (torch 2.10, box64 v0.4.5): **3.75 ms on the BPU vs 72.06 ms
eager CPU — 19.2x.** An earlier measurement on a slightly different stack gave
4.00 ms vs 94.35 ms (23.6x) with cosine similarity 0.981; the ratio depends on
how much of the graph is convolution.

On a small 2-conv net the relative error against eager float is ~3%, which is the
expected cost of int8 activations — set `FLAGOS_BPU_QUANTIZE=0` for a bit-exact
float artifact, at the price of the conv work falling back to the CPU.

Toy networks are a wash (0.849 ms vs 0.805 ms) — the fixed submission cost
dominates, and there is not enough MAC work to amortize it. The offload is worth
it when the graph is genuinely convolution-heavy.

## Known limitations

- Single device only; the four BPU cores are not scheduled independently.
- Synchronous execution; `hbDNNInferAsync` is unused.
- int8 only. hbdk4 also supports si16, which would trade throughput for accuracy.
- The zero-copy path is partial. `runtime.py` wraps a flagos tensor's UCP
  storage in a numpy array in place (`_device_view`), so there is no
  device-to-host copy on the way in — but quantization changes dtype (float32 in
  the graph, int8 in the artifact), and that conversion copies. Only a
  dtype-matched input is truly copy-free. Outputs always copy: `hbm_runtime.run`
  allocates its own arrays. Driving `hbDNNInferV2` directly with tensors built
  from `FlagosBPUPhysicalAddress()` would close the remaining gap.
- Calibration is opt-in and manual.
