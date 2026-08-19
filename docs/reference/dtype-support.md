# Dtype Support

FlagOS preserves the requested dtype for tensor storage and follows PyTorch's
promotion rules for tensor-tensor operations. Compute coverage is still bounded
by the vendor library used by each backend. A dtype appearing in a backend's
storage table does not imply that every operator accepts it.

## AMP

`torch.autocast("flagos")` supports `torch.float16` and `torch.bfloat16` as
lower-precision targets. The standard PyTorch autocast policy groups are used:
matmul and convolution prefer the selected lower-precision dtype, numerical
operations such as logarithm and normalization use float32, and mixed inputs
follow the promote policy. Float32 and float64 are not valid autocast targets.

AMP target support is separate from eager dtype support. For example, Ascend
can store and perform many elementwise operations on float64 even though float64
is not an AMP target and is not accepted by its native matmul API.

## Backend Matrix

| Backend | Storage and copy | Eager elementwise | Matmul family | AMP targets |
| --- | --- | --- | --- | --- |
| CUDA | Native PyTorch CUDA dtype support | Native CUDA coverage | Native CUDA coverage | float16, bfloat16 |
| Ascend | float16, bfloat16, float32, float64, integer, uint8, bool | Vendor coverage; unsupported ACLNN combinations use the CPU fallback | float16, bfloat16, float32 natively; float64 and unsupported types use the CPU fallback | float16, bfloat16 |
| MetaX | Vendor library coverage | Vendor library coverage | Vendor library coverage | Backend-specific |
| DCU | Vendor library coverage | Vendor library coverage | Vendor library coverage | Backend-specific |
| MUSA | Vendor library coverage | Vendor library coverage | Vendor library coverage | Backend-specific |

The MetaX, DCU, and MUSA entries intentionally do not claim parity without
hardware measurements for the specific library release. Their native boxing or
code-generated kernels determine the available operator/dtype combinations.

## Ascend Boundaries

Ascend CANN 9.0 exposes the ACL dtype enums for float64 and all of the common
integer types, but individual ACLNN operators have narrower contracts:

- `aclnnMatmul`, `aclnnMm`, and `aclnnBatchMatMul` reject float64 and integer
  inputs. FlagOS uses the existing CPU fallback and copies the correctly typed
  result back to the Ascend device.
- `aclnnNeg` rejects int16, uint8, and bool. FlagOS uses the CPU reference for
  those inputs, preserving PyTorch's integer wraparound behavior and bool error.
- Ascend float64 storage, device copies, casts, and elementwise operations remain
  float64. The implementation must not clamp float64 to float32 during `_to_copy`.
- Complex and quantized dtypes are not currently mapped by the Ascend ACL tensor
  wrapper and are outside the supported contract.

The fallback is correctness-oriented and may be slower than a native kernel.
New Ascend operator routes must be added through `scripts/codegen_ascend.py` and
regenerated; generated output is not an independent source of truth.

## Testing Contract

The integration tests cover factory, unary, binary, reduction, indexing,
comparison, copy, promotion, AMP, and GradScaler behavior. Dtype tests compare
results with CPU references where a native Ascend kernel is unavailable. A
passing test means the operation follows the documented PyTorch contract, not
that the operation necessarily uses a native vendor kernel.
