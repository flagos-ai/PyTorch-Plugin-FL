// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <limits>
#include <ATen/Dispatch.h>
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
inline mcblasHandle_t GetMcblasHandleForBmm() {
  thread_local mcblasHandle_t handle = nullptr;
  if (handle == nullptr) {
    const mcblasStatus_t status = mcblasCreate(&handle);
    TORCH_CHECK(
        status == MCBLAS_STATUS_SUCCESS,
        "MetaX bmm: mcblasCreate failed: ",
        mcblasGetStatusString(status));
  }
  return handle;
}

template <typename scalar_t>
struct BmmGemmExConfig;

template <>
struct BmmGemmExConfig<float> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_32F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F;
};

template <>
struct BmmGemmExConfig<double> {
  using alpha_t = double;
  static constexpr macaDataType type = MACA_R_64F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_64F;
};

template <>
struct BmmGemmExConfig<at::Half> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_16F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_FAST_16F;
};

template <>
struct BmmGemmExConfig<at::BFloat16> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_16BF;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_FAST_16BF;
};

template <typename scalar_t>
void LaunchBmmMcblas(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  const at::Tensor a = self.contiguous();
  const at::Tensor b = mat2.contiguous();
  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b.size(2);
  TORCH_CHECK(b.size(0) == batch && b.size(1) == k, "MetaX bmm: shape mismatch");
  TORCH_CHECK(
      batch <= std::numeric_limits<int>::max() &&
          m <= std::numeric_limits<int>::max() &&
          n <= std::numeric_limits<int>::max() &&
          k <= std::numeric_limits<int>::max(),
      "MetaX bmm: shape too large for mcblasGemmStridedBatchedEx int32 dimensions");

  at::Tensor out_work = out;
  if (!out.is_contiguous()) {
    out_work = at::empty(
        out.sizes(),
        out.options().memory_format(at::MemoryFormat::Contiguous));
  }

  const auto handle = GetMcblasHandleForBmm();
  const mcblasStatus_t set_stream_status =
      mcblasSetStream(handle, reinterpret_cast<mcStream_t>(metax::CurrentStream()));
  TORCH_CHECK(
      set_stream_status == MCBLAS_STATUS_SUCCESS,
      "MetaX bmm: mcblasSetStream failed: ",
      mcblasGetStatusString(set_stream_status));

  using cfg = BmmGemmExConfig<scalar_t>;
  const typename cfg::alpha_t alpha = static_cast<typename cfg::alpha_t>(1);
  const typename cfg::alpha_t beta = static_cast<typename cfg::alpha_t>(0);

  // Row-major batched A(batch,m,k) * B(batch,k,n) -> column-major mapping:
  // C_col(batch,n,m) = B_col(batch,n,k) * A_col(batch,k,m)
  const int m_col = static_cast<int>(n);
  const int n_col = static_cast<int>(m);
  const int k_col = static_cast<int>(k);
  const int lda = static_cast<int>(n);
  const int ldb = static_cast<int>(k);
  const int ldc = static_cast<int>(n);
  const mcblas_stride stride_a = static_cast<mcblas_stride>(n * k);
  const mcblas_stride stride_b = static_cast<mcblas_stride>(k * m);
  const mcblas_stride stride_c = static_cast<mcblas_stride>(n * m);

  mcblasStatus_t status = MCBLAS_STATUS_SUCCESS;
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    at::Half alpha_h = at::Half(1.0f);
    at::Half beta_h = at::Half(0.0f);
    status = mcblasHgemmStridedBatched(
        handle,
        MCBLAS_OP_N,
        MCBLAS_OP_N,
        m_col,
        n_col,
        k_col,
        reinterpret_cast<const mcblas_half*>(&alpha_h),
        reinterpret_cast<const mcblas_half*>(b.data_ptr<at::Half>()),
        lda,
        stride_a,
        reinterpret_cast<const mcblas_half*>(a.data_ptr<at::Half>()),
        ldb,
        stride_b,
        reinterpret_cast<const mcblas_half*>(&beta_h),
        reinterpret_cast<mcblas_half*>(out_work.data_ptr<at::Half>()),
        ldc,
        stride_c,
        static_cast<int>(batch));
  } else {
    status = mcblasGemmStridedBatchedEx(
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
        stride_a,
        a.data_ptr<scalar_t>(),
        cfg::type,
        ldb,
        stride_b,
        &beta,
        out_work.data_ptr<scalar_t>(),
        cfg::type,
        ldc,
        stride_c,
        static_cast<int>(batch),
        cfg::compute_type,
        MCBLAS_GEMM_DEFAULT);
  }
  TORCH_CHECK(
      status == MCBLAS_STATUS_SUCCESS,
      "MetaX bmm: mcblas call failed: ",
      mcblasGetStatusString(status));

  if (!out.is_contiguous()) {
    out.copy_(out_work);
  }
}
#endif

} // namespace

inline void BmmKernelMetax(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(
      self.dim() == 3 && mat2.dim() == 3, "MetaX bmm: inputs must be 3-D");
  TORCH_CHECK(
      self.scalar_type() == mat2.scalar_type(),
      "MetaX bmm: expected self and mat2 to have the same dtype");
  at::native::resize_output(out, {self.size(0), self.size(1), mat2.size(2)});
  TORCH_CHECK(
      out.scalar_type() == self.scalar_type(),
      "MetaX bmm: expected out to have the same dtype as inputs");

  if (out.numel() == 0) {
    return;
  }

#if FLAGOS_METAX_HAS_MCBLAS
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "bmm_metax",
      [&]() { LaunchBmmMcblas<scalar_t>(self, mat2, out); });
#else
  TORCH_CHECK(false, "MetaX bmm requires mcblas headers and library");
#endif
}

} // namespace at::native::flagos
