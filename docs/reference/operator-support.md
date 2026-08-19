# Operator Support

This reference records measured operator coverage for torch-fl accelerator
backends. The current baseline measures the generic FlagGems Python routing
surface on four hardware platforms. It is an availability and correctness
survey, not a claim of complete PyTorch conformance, autograd coverage, or
performance quality.

The measurement unit is an active, unique, exact ATen overload such as
`sum.dim_IntList`. It is different from an OpInfo base operation, so historical
OpInfo totals such as 158 must not be compared with the 546-overload denominator
below.

Routing-table presence alone is not proof that an overload executes correctly.
Conversely, an overload without a direct route may still execute through a
composite decomposition or fallback. See the [Compatibility Matrix](compatibility.md),
[unrouted operator analysis](../vendors/flaggems/unrouted-ops.md), and
[no-dispatcher analysis](../vendors/flaggems/no-dispatcher-analysis.md) for those
separate concerns.

## Verdicts

The manual survey first rejects synthesized invocations that are invalid on the
CPU reference. It then classifies each overload from the remaining valid cases:

| Verdict | Definition |
|---|---|
| `STRICT` | Every CPU-valid synthesized case passed on the target hardware. |
| `BASIC_ONLY` | At least one CPU-valid case passed, but one or more other valid cases failed. |
| `FAILED` | Valid cases existed and none passed. |
| `UNTESTED` | No CPU-valid synthesized case existed; this is neither a pass nor a failure. |

**Basic executable** is `STRICT + BASIC_ONLY`.

`PASS`, `INVALID_CASE`, `UNVERIFIABLE`, `ERROR`, `WRONG`, `CRASH`, and
`TIMEOUT` are case-level statuses, not additional operator verdicts.
`INVALID_CASE` and `UNVERIFIABLE` are excluded from support classification.

## Baseline Cohort

All hardware rows in this baseline use the same active route set and survey
methodology. These revisions identify the measured cohort; they do not describe
the current repository HEAD.

| Field | Value |
|---|---|
| torch-fl source | `fe2272b5fd1313eff00017c3f8242afe6c9a2cf6` |
| FlagGems source | `7fb49bad47116434961bfb2b912811716d383eaf` |
| Generic config | `torch_fl/configs/backends_flaggems.conf` |
| Generic config SHA-256 | `f97686deec8aa4863ecd04d359960804cbdf5862d27449e6345e3451512db9d8` |
| Active route-set SHA-256 | `8a1649e79ef7c419c050d65465c46dcf25575303c74d61dc194c5838ea847456` |
| Survey harness | `tests/manual/flaggems_overload_survey.py`, version 4 |
| Survey harness SHA-256 | `2354d4f76a6b37831492979dae25b9318cbe94fb48e08cdf100a4cab09cebd13` |
| FlagGems `_FULL_CONFIG` entries | 866 |
| Generated Python routes | 572 |
| Active surveyed routes | 546 |
| Forced CUDA fallbacks | 26 |
| Profiles per overload | 7 |

Full generation discovers 572 Python routes. The generic production
configuration activates 546 as `flagos_python` and forces 26 to CUDA fallback,
which explains the 546-route survey denominator.

## Hardware Summary

Rates use all 546 active routes as the denominator and are rounded to one
decimal place.

| Hardware | Total | STRICT | BASIC_ONLY | FAILED | UNTESTED | Basic executable | Basic rate | Strict rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 546 | 348 | 54 | 46 | 98 | 402 | 73.6% | 63.7% |
| MetaX mc550 | 546 | 260 | 33 | 155 | 98 | 293 | 53.7% | 47.6% |
| PPU 810e | 546 | 347 | 54 | 47 | 98 | 401 | 73.4% | 63.6% |
| Hygon DCU bw1000 | 546 | 321 | 53 | 74 | 98 | 374 | 68.5% | 58.8% |

For every row, `STRICT + BASIC_ONLY + FAILED + UNTESTED = Total`, and
`Basic executable = STRICT + BASIC_ONLY`.

## Raw Case Evidence

These counts cover seven synthesized profiles per overload. They are case-level
data and therefore do not share the 546-overload denominator of the hardware
summary.

| Hardware | PASS | INVALID_CASE | UNVERIFIABLE | ERROR | WRONG | CRASH | TIMEOUT | Context poison |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 2163 | 1309 | 0 | 184 | 152 | 14 | 0 | 0 |
| MetaX mc550 | 1597 | 1309 | 0 | 812 | 90 | 14 | 0 | 0 |
| PPU 810e | 2158 | 1305 | 0 | 184 | 154 | 21 | 0 | 0 |
| Hygon DCU bw1000 | 2016 | 1309 | 0 | 337 | 146 | 14 | 0 | 0 |

The bw1000 baseline excludes 108 initial records that failed before operator
execution because the child process could not load its MPI runtime. Exactly
those routes were rerun with the correct runtime environment; the corrected
result has 14 remaining `CRASH` cases, all return code `-11`.

## Reproducing and Updating the Report

Run the manual survey on each target hardware platform:

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_flaggems.conf \
  --out /tmp/flaggems-overloads.json
```

When operator routing or implementation changes:

1. Run from an identified torch-fl revision with an identified FlagGems
   revision on every affected hardware platform.
2. Record the exact hardware model, run date, source revisions, configuration
   SHA-256, active route-set SHA-256, and harness version/SHA-256.
3. Keep the active route set fixed for cross-hardware comparisons. If cohorts
   differ, label that difference explicitly rather than presenting the rows as
   directly comparable.
4. Recompute the four overload verdicts centrally from raw cases. Do not treat
   `CRASH` or `TIMEOUT` as operator verdicts and do not infer support from a
   configured route.
5. Update both tables, verify their arithmetic, and append an update-history
   entry describing the affected hardware and evidence.
6. If hardware is unavailable, mark the affected row **not revalidated** and
   document the evidence gap in this report and the PR.

Keep the per-overload JSON as the auditable evidence. Do not expand this report
into a 546-row inventory; the aggregate tables are the maintained human-facing
record.

## Native Backend Route Changes

The generic FlagGems survey above does not exercise vendor-native routes such as
`ascend` or `gcu`. Native route changes are tracked here separately so they are
not misrepresented as part of the 546-overload FlagGems cohort.

### Enflame GCU S60 RNG routes (2026-08-17)

The GCU backend added native topsaten routes for the following RNG overloads:

- `bernoulli`, `bernoulli_.float`
- `exponential`, `exponential_`
- `multinomial`
- `poisson`
- `randn`, `randn.generator`
- `randn_like.generator`, `randn_like.generator_out`
- `randint.generator`, `randint.low_generator`
- `randperm.generator`
- `random_`, `random_.to`

Generator-less calls on these routes consume the same explicit topsaten
`{seed, offset}` stream used by FlagGems; explicit generators remain isolated.
Unsupported dtypes continue through the CPU fallback.

Targeted validation ran on an Enflame S60 with the installed TopsRider SDK:

- `tests/integration/ops/test_rng_dispatch.py`: `104 passed, 2 skipped, 1 xpassed`.
- Mixed route probe with `randn -> flagos_python` and `exponential_ -> gcu`:
  shared state advanced `(1234, 0) -> (1234, 8) -> (1234, 40)`; same-seed
  replay, different-seed sensitivity, and mixed-state replay all passed.

The standard `flaggems_overload_survey.py` harness is not applicable to these
native routes because it selects only `flagos_python` overloads. This targeted
RNG evidence does not revalidate the separate generic FlagGems support cohort.

### Ascend FSDP2 routes (2026-08-14)

The Ascend backend added or enabled the following FSDP2 paths:

- `_chunk_cat`
- `_chunk_cat.out`
- `_foreach_copy_`
- `cat.out`
- `split.Tensor`
- `split_with_sizes`
- `split_with_sizes_copy.out`

The standard `flaggems_overload_survey.py` harness cannot measure these routes:
it deliberately selects only `flagos_python` entries. Instead, these native
routes were exercised end-to-end on two physical Ascend 910 devices with CANN
9.0 and `ASCEND_RT_VISIBLE_DEVICES=2,3`:

- FlagCX collective test: passed all-reduce, broadcast, all-gather,
  reduce-scatter, and barrier.
- DDP test: passed forward, backward, gradient synchronization, and optimizer
  step (final losses `0.061326` and `0.118651`).
- FSDP2 test: passed parameter all-gather, gradient reduce-scatter, forward,
  backward, and optimizer step; each rank produced four finite gradient tensors
  (final losses `0.044212` and `0.063512`).

The generic FlagGems rows are **not revalidated** by this change because their
active route cohort is unchanged. The evidence gap is that there is no
per-overload synthesized survey for vendor-native Ascend routes; the available
evidence is the targeted FSDP2/DDP/collective workload described above.

### MUSA native RNG routes (2026-08-17)

The MUSA route configuration includes native muRAND/mudnn implementations for the core RNG families (`rand`, `randn`, `rand_like`, `randn_like`, `randint`, `normal_`, `uniform_`, `random_`, and native dropout). They share the authoritative per-device PrivateUse1 generator with the optional FlagGems Philox bridge. `randperm` and unsupported distribution overloads remain on CPU fallback and are not counted as native support.

These native routes were measured on an eight-device Moore Threads MTT S5000 host. Device 0 reported capability 3.1, 60 multiprocessors, and 85,813,358,592 bytes of memory. With CPU PyTorch 2.10.0 and the installed `/usr/local/musa` toolkit (`mudnn` v3300):

- `tests/integration/ops/test_musa_rng.py`: **7 passed**. Coverage includes same-seed reproducibility, `torch.manual_seed`, `torch.flagos.manual_seed`/`manual_seed_all`, state round trips, explicit generators, integer/out/like variants, full-width int64 ranges, `[0, 1)` uniform bounds, native dropout forward/backward, shared native/FlagGems reservation ordering, and per-device sequence isolation.
- `tests/integration/ops/test_musa_dispatch.py`: **89 passed**.
- `tests/unit/test_vendor_routing.py` plus `tests/unit/test_musa_rng_bridge.py`: **24 passed**.

The target cohort is the available MTT S5000 host; no S6000 claim is made.

The MUSA hybrid config adds seven non-overlapping FlagGems Python routes (`all`, `all.dims`, `any`, `any.dims`, `index_add`, `index_add_`, and `repeat_interleave.Tensor`) while retaining native RNG precedence. They were execution-validated with FlagGems 5.0.2 and the vendor `flagtree-0.5.0+mthreads3.1` wheel (Triton 3.1.0, backend `mthreads`; SHA-256 `197b0c6954ad8b3edef51138311a8c4f3aea75b90ba0f69d3c2fda95a76b6b1b`). `tests/integration/ops/test_musa_flaggems.py` passed **2 tests in 5.33 seconds** on `flagos:0`: instrumentation observed every configured wrapper, it compares selected route outputs against CPU, includes duplicate-index `index_add`, checks in-place `index_add_`, and launches FlagGems `randn` on `flagos:0` between native `rand` calls. Repeating after `torch.flagos.manual_seed(20260817)` reproduced all outputs and confirmed the two shared C++ generator reservations. Native and hybrid suites must run in separate pytest processes because the C++ `BackendTable()` caches the backend configuration on first use. The generic installed Triton 3.7.1 is not MThreads-capable and is not execution evidence.

## Update History

| Date | Hardware | Cohort | Change | Evidence |
|---|---|---|---|---|
| 2026-08-18 | MTT S5000 (8 devices) | Native MUSA RNG, MThreads FlagGems hybrid, and MUPTI profiler | Added optional MUPTI activity tracing; the operator route cohort is unchanged. | `tests/integration/test_profiler_musa.py`: 1 passed with real positive-duration MUPTI kernel/runtime/memcpy activities and valid Chrome JSON. CPU-only Kineto resolver behavior remains environment-dependent; generic FlagGems operator coverage was not revalidated by this profiler change. |
| 2026-08-17 | MTT S5000 (8 devices) | Native MUSA RNG and MThreads FlagGems hybrid | Added shared per-device RNG reservations, muRAND/mudnn native RNG, shared stream compatibility, and seven non-overlapping FlagGems routes. | Native RNG: 7 passed; MUSA dispatch: 89 passed; routing/bridge units: 24 passed; real hybrid FlagGems: 2 passed, including selected reductions, duplicate-index `index_add`, and FlagGems `randn` mixed with native RNG. Vendor FlagTree wheel required; generic Triton 3.7.1 is not evidence. |
| 2026-08-17 | Enflame S60 | Native GCU RNG routes | Added 16 topsaten RNG routes; generic FlagGems cohort not revalidated. | Targeted mixed native/FlagGems probe verified shared seed/offset progression and replay; `tests/integration/ops/test_rng_dispatch.py`: `104 passed, 2 skipped, 1 xpassed`. |
| 2026-08-14 | Ascend 910 (2 devices) | Native Ascend FSDP2 routes | Added `_chunk_cat`, `_chunk_cat.out`, `_foreach_copy_`, `cat.out`, `split.Tensor`, `split_with_sizes`, and `split_with_sizes_copy.out`; generic FlagGems cohort not revalidated because it is unchanged. | Manual FlagCX collective, DDP, and FSDP2 tests on CANN 9.0; standard FlagGems harness is not applicable to native routes. |
| 2026-08-13 | A100, mc550, 810e, bw1000 | torch-fl `fe2272b5`, FlagGems `7fb49bad`, harness v4 | Established the verified 546-overload four-platform baseline. | Manual survey JSON; aggregate and raw counts recorded above. |
