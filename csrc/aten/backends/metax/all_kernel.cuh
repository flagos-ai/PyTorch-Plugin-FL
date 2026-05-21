// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

__global__ void AllBoolKernel(int64_t n, const bool* data, int* flag) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (!data[idx]) {
      atomicExch(flag, 0);
      return;
    }
  }
}

int LaunchAllBoolReduce(int64_t n, const bool* data) {
  int* device_flag = nullptr;
  TORCH_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&device_flag), sizeof(int)) ==
          cudaSuccess,
      "MetaX all: cudaMalloc failed");
  TORCH_CHECK(
      cudaMemset(device_flag, 1, sizeof(int)) == cudaSuccess,
      "MetaX all: cudaMemset failed");

  const int blocks = static_cast<int>(
      (n + metax::kBlockSize - 1) / metax::kBlockSize);
  AllBoolKernel<<<blocks, metax::kBlockSize, 0, metax::CurrentStream()>>>(
      n, data, device_flag);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "MetaX all kernel launch failed");

  int host_flag = 0;
  TORCH_CHECK(
      cudaMemcpy(
          &host_flag,
          device_flag,
          sizeof(int),
          cudaMemcpyDeviceToHost) == cudaSuccess,
      "MetaX all: cudaMemcpy failed");
  cudaFree(device_flag);
  return host_flag;
}

}  // namespace

inline at::Tensor AllKernel(const at::Tensor& self) {
  at::Tensor result = at::empty({}, self.options().dtype(at::kBool));

  if (self.numel() == 0) {
    result.fill_(true);
    return result;
  }

  const at::Tensor input = self.contiguous().to(at::kBool);
  const int64_t n = input.numel();
  const bool* data = input.const_data_ptr<bool>();

  result.fill_(LaunchAllBoolReduce(n, data) != 0);
  return result;
}

}  // namespace at::native::flagos
