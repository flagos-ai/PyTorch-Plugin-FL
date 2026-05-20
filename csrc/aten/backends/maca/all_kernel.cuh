// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include "maca_elementwise.cuh"

namespace at::native::flagos {

namespace {

__global__ void all_bool_kernel(int64_t n, const bool* data, int* flag) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (!data[idx]) {
      atomicExch(flag, 0);
      return;
    }
  }
}

} // namespace

inline at::Tensor AllKernel(const at::Tensor& self) {
  at::Tensor result = at::empty({}, self.options().dtype(at::kBool));

  if (self.numel() == 0) {
    result.fill_(true);
    return result;
  }

  at::Tensor input = self.contiguous().to(at::kBool);
  const int64_t n = input.numel();
  const bool* data = input.const_data_ptr<bool>();

  int* d_flag = nullptr;
  TORCH_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_flag), sizeof(int)) == cudaSuccess,
      "MACA all: cudaMalloc failed");
  TORCH_CHECK(
      cudaMemset(d_flag, 1, sizeof(int)) == cudaSuccess,
      "MACA all: cudaMemset failed");

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  all_bool_kernel<<<blocks, threads, 0, maca::current_stream()>>>(n, data, d_flag);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "MACA all kernel launch failed");

  int h_flag = 0;
  TORCH_CHECK(
      cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
      "MACA all: cudaMemcpy failed");
  cudaFree(d_flag);

  result.fill_(h_flag != 0);
  return result;
}

} // namespace at::native::flagos
