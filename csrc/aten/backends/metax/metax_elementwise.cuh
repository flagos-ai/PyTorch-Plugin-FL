// Copyright (c) 2026, BAAI. All rights reserved.
//
// Minimal elementwise launch for MetaX (mxcc/cucc + cu-bridge runtime).
// Does not use PyTorch CUDALoops or at::cuda::*.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include <c10/util/Exception.h>

namespace at::native::flagos::metax {

constexpr int kBlockSize = 256;

inline cudaStream_t CurrentStream() {
  return nullptr;
}

template <typename Kernel, typename... Args>
inline void Launch1d(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>(
      (n + static_cast<int64_t>(kBlockSize) - 1) / kBlockSize);
  kernel<<<blocks, kBlockSize, 0, CurrentStream()>>>(n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX kernel launch failed: ",
      cudaGetErrorString(err));
}

} // namespace at::native::flagos::metax
