// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device copy for the MUSA backend: the replacement for torch_musa's
// `at::musa::_copy_from`.
//
// mudnn's Unary ops read strides on both operands, which was verified directly
// against the library: a transposed source gathers correctly into a contiguous
// destination, a strided destination is scattered into without touching the gaps,
// stride-0 (broadcast) sources replicate, and CAST does any of those while
// converting dtype. So every copy shape this backend needs is one Unary::Run --
// no temporary buffer, no CPU round-trip.

#include "aten/backends/musa/mudnn_common.h"

#ifdef USE_MUSA

namespace at::native::flagos::musa_ops {

void MudnnCopy(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(
      src.defined() && dst.defined(), "MudnnCopy: undefined tensor");
  if (dst.numel() == 0) {
    return;
  }
  TORCH_CHECK(
      MudnnSupportsDtype(src.scalar_type()) &&
          MudnnSupportsDtype(dst.scalar_type()),
      "MudnnCopy: unsupported dtype ", src.scalar_type(), " -> ",
      dst.scalar_type());

  // Broadcast to the destination shape when needed. expand() only adjusts
  // sizes/strides (introducing 0 strides), which mudnn handles, so this stays a
  // view -- no allocation.
  at::Tensor src_view = src;
  if (!src.sizes().equals(dst.sizes())) {
    src_view = src.expand(dst.sizes());
  }

  MudnnTensorWrapper t_src(src_view);
  MudnnTensorWrapper t_dst(dst);

  mudnn::Unary op;
  const bool needs_cast = src.scalar_type() != dst.scalar_type();
  op.SetMode(
      needs_cast ? mudnn::Unary::Mode::CAST : mudnn::Unary::Mode::IDENTITY);

  EXEC_MUDNN_CMD(
      needs_cast ? "mudnn copy (CAST)" : "mudnn copy (IDENTITY)",
      dst,
      op.Run(_mudnn_h, t_dst.get(), t_src.get()));
}

} // namespace at::native::flagos::musa_ops

#endif // USE_MUSA
