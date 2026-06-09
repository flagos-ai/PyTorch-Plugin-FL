// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>
#include <c10/util/BFloat16.h>

#include "metax_elementwise.cuh"

#if __has_include(<mcblas/mcblas.h>)
#include <mcblas/mcblas.h>
#define FLAGOS_METAX_HAS_MCBLAS 1
#else
#define FLAGOS_METAX_HAS_MCBLAS 0
#endif

namespace at::native::flagos {

namespace {

#if FLAGOS_METAX_HAS_MCBLAS
inline mcblasHandle_t GetMcblasHandle() {
  thread_local mcblasHandle_t handle = nullptr;
  if (handle == nullptr) {
    const mcblasStatus_t status = mcblasCreate(&handle);
    TORCH_CHECK(
        status == MCBLAS_STATUS_SUCCESS,
        "MetaX mm: mcblasCreate failed: ",
        mcblasGetStatusString(status));
  }
  return handle;
}

template <typename scalar_t>
struct GemmExConfig;

template <>
struct GemmExConfig<float> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_32F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F;
};

template <>
struct GemmExConfig<double> {
  using alpha_t = double;
  static constexpr macaDataType type = MACA_R_64F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_64F;
};

template <>
struct GemmExConfig<at::Half> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_16F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_FAST_16F;
};

template <>
struct GemmExConfig<at::BFloat16> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_16BF;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_FAST_16BF;
};

template <typename scalar_t>
void LaunchMmMcblas(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  const at::Tensor a = self.contiguous();
  const at::Tensor b = mat2.contiguous();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  TORCH_CHECK(b.size(0) == k, "MetaX mm: shape mismatch");
  TORCH_CHECK(
      m <= std::numeric_limits<int>::max() &&
          n <= std::numeric_limits<int>::max() &&
          k <= std::numeric_limits<int>::max(),
      "MetaX mm: shape too large for mcblasGemmEx int32 dimensions");

  at::Tensor out_work = out;
  if (!out.is_contiguous()) {
    out_work = at::empty(
        out.sizes(),
        out.options().memory_format(at::MemoryFormat::Contiguous));
  }

  const auto handle = GetMcblasHandle();
  const mcblasStatus_t set_stream_status =
      mcblasSetStream(handle, reinterpret_cast<mcStream_t>(metax::CurrentStream()));
  TORCH_CHECK(
      set_stream_status == MCBLAS_STATUS_SUCCESS,
      "MetaX mm: mcblasSetStream failed: ",
      mcblasGetStatusString(set_stream_status));

  using cfg = GemmExConfig<scalar_t>;
  const typename cfg::alpha_t alpha = static_cast<typename cfg::alpha_t>(1);
  const typename cfg::alpha_t beta = static_cast<typename cfg::alpha_t>(0);

  // Row-major A(m,k) * B(k,n) can be computed via column-major:
  // C_col(n,m) = B_col(n,k) * A_col(k,m)
  const int m_col = static_cast<int>(n);
  const int n_col = static_cast<int>(m);
  const int k_col = static_cast<int>(k);
  const int lda = static_cast<int>(n);
  const int ldb = static_cast<int>(k);
  const int ldc = static_cast<int>(n);

  mcblasStatus_t status = MCBLAS_STATUS_SUCCESS;
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    at::Half alpha_h = at::Half(1.0f);
    at::Half beta_h = at::Half(0.0f);
    status = mcblasHgemm(
        handle,
        MCBLAS_OP_N,
        MCBLAS_OP_N,
        m_col,
        n_col,
        k_col,
        reinterpret_cast<const mcblas_half*>(&alpha_h),
        reinterpret_cast<const mcblas_half*>(b.data_ptr<at::Half>()),
        lda,
        reinterpret_cast<const mcblas_half*>(a.data_ptr<at::Half>()),
        ldb,
        reinterpret_cast<const mcblas_half*>(&beta_h),
        reinterpret_cast<mcblas_half*>(out_work.data_ptr<at::Half>()),
        ldc);
  } else {
    status = mcblasGemmEx(
        handle,
        MCBLAS_OP_N,
        MCBLAS_OP_N,
        m_col,
        n_col,
        k_col,
        &alpha,
        b.data_ptr<scalar_t>(),
        cfg::type,
        lda,
        a.data_ptr<scalar_t>(),
        cfg::type,
        ldb,
        &beta,
        out_work.data_ptr<scalar_t>(),
        cfg::type,
        ldc,
        cfg::compute_type,
        MCBLAS_GEMM_DEFAULT);
  }

  TORCH_CHECK(
      status == MCBLAS_STATUS_SUCCESS,
      "MetaX mm: mcblas call failed: ",
      mcblasGetStatusString(status));

  if (!out.is_contiguous()) {
    out.copy_(out_work);
  }
}
#endif

} // namespace

inline void MmKernelMetax(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "MetaX mm: inputs must be 2-D");
  TORCH_CHECK(
      self.scalar_type() == mat2.scalar_type(),
      "MetaX mm: expected self and mat2 to have the same dtype");
  at::native::resize_output(out, {self.size(0), mat2.size(1)});
  TORCH_CHECK(
      out.scalar_type() == self.scalar_type(),
      "MetaX mm: expected out to have the same dtype as inputs");

  if (out.numel() == 0) {
    return;
  }

#if FLAGOS_METAX_HAS_MCBLAS
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "mm_metax",
      [&]() { LaunchMmMcblas<scalar_t>(self, mat2, out); });
#else
  TORCH_CHECK(false, "MetaX mm requires mcblas headers and library");
#endif
}

} // namespace at::native::flagos
