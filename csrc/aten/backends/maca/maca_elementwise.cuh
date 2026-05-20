// Copyright (c) 2026, BAAI. All rights reserved.
//
// Minimal elementwise launch for MACA (mxcc/cucc + cu-bridge runtime).
// Does not use PyTorch CUDALoops or at::cuda::*.

#pragma once

#include <cuda_runtime.h>

#include <c10/util/Exception.h>

namespace at::native::flagos::maca {

inline cudaStream_t current_stream() {
  return nullptr;
}

template <typename Kernel, typename... Args>
inline void launch_1d(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int threads = 256;
  const int blocks = static_cast<int>(
      (n + static_cast<int64_t>(threads) - 1) / threads);
  kernel<<<blocks, threads, 0, current_stream()>>>(n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MACA kernel launch failed: ",
      cudaGetErrorString(err));
}

} // namespace at::native::flagos::maca
