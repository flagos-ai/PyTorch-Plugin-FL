// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device strided copy for the Ascend backend. Lets the platform-neutral
// copy_/clone/contiguous paths avoid the CPU round-trip (device->host strided
// copy->device) that dominates GQA repeat_kv clones in Qwen3 inference.

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::flagos::ascend {

// Copy `src` into `dst` entirely on the NPU via aclnnInplaceCopy, which handles
// differing strides/offsets and dtype casts on-device. `dst` must be an
// allocated PrivateUse1 tensor with matching sizes; `src` may be non-contiguous.
// Returns true on success. Callers use the return value to fall back to the CPU
// round-trip if the on-device path is unavailable.
bool StridedCopy(const at::Tensor& dst, const at::Tensor& src);

// Cast `src` (a contiguous PrivateUse1 tensor) to `dtype` entirely on the NPU
// via aclnnCast, returning a freshly-allocated contiguous PrivateUse1 tensor.
// Replaces the D2H -> CPU cast -> H2D round-trip in _to_copy's Ascend dtype
// path, which dominated HF RMSNorm (two fp16<->fp32 casts per layer). Returns
// an undefined tensor if the on-device path is unavailable (caller falls back).
at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype);

} // namespace at::native::flagos::ascend
