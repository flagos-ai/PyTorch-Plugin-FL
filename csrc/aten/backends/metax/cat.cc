// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cat.h"

#include <include/flagos.h>

#include <vector>

namespace at::native::flagos {

namespace {

at::Tensor CatKernelMetax(const at::ITensorListRef& tensors, int64_t dim) {
  std::vector<at::Tensor> materialized;
  materialized.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    materialized.push_back(tensor.contiguous());
  }
  TORCH_CHECK(!materialized.empty(), "MetaX cat: expected a non-empty TensorList");

  int64_t ref_idx = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(materialized.size()); ++i) {
    // Match PyTorch cat semantics: skip 1D empty tensors when inferring shape/dim.
    if (!(materialized[i].dim() == 1 && materialized[i].numel() == 0)) {
      ref_idx = i;
      break;
    }
  }
  if (ref_idx < 0) {
    at::Tensor out = materialized[0].clone();
    return out;
  }
  const auto& first = materialized[ref_idx];
  dim = at::maybe_wrap_dim(dim, first.dim());
  TORCH_CHECK(
      dim < first.dim(),
      "MetaX cat: dimension ",
      dim,
      " out of range");

  int64_t out_dim_size = 0;
  for (const auto& tensor : materialized) {
    if (tensor.dim() == 1 && tensor.numel() == 0) {
      continue;
    }
    TORCH_CHECK(
        tensor.scalar_type() == first.scalar_type(),
        "MetaX cat: all inputs must have the same dtype");
    TORCH_CHECK(
        tensor.device() == first.device(),
        "MetaX cat: all inputs must be on the same device");
    for (int64_t d = 0; d < first.dim(); ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_CHECK(
          tensor.size(d) == first.size(d),
          "MetaX cat: sizes must match except in the concatenation dimension");
    }
    out_dim_size += tensor.size(dim);
  }

  auto out_sizes = first.sizes().vec();
  out_sizes[dim] = out_dim_size;
  at::Tensor out = at::empty(out_sizes, first.options());

  if (out.numel() == 0) {
    return out;
  }

  // For contiguous tensors, each fixed index over all dimensions except `dim`
  // owns a run of `size(dim) * inner_size` elements, where inner_size is the
  // product of sizes after `dim` (e.g. head_dim=128 when cat on seq dim).
  int64_t inner_size = 1;
  for (int64_t d = dim + 1; d < first.dim(); ++d) {
    inner_size *= first.size(d);
  }
  const int64_t outer_size =
      first.numel() / (first.size(dim) * inner_size);
  const size_t elem_size = static_cast<size_t>(first.element_size());

  for (int64_t o = 0; o < outer_size; ++o) {
    int64_t offset_along_dim = 0;
    for (const auto& tensor : materialized) {
      if (tensor.dim() == 1 && tensor.numel() == 0) {
        continue;
      }
      const int64_t chunk = tensor.size(dim);
      const int64_t slice_elems = chunk * inner_size;
      if (slice_elems == 0) {
        offset_along_dim += chunk;
        continue;
      }

      const size_t nbytes = static_cast<size_t>(slice_elems) * elem_size;
      const char* src = static_cast<const char*>(tensor.data_ptr()) +
          static_cast<size_t>(o * chunk * inner_size) * elem_size;
      char* dst = static_cast<char*>(out.data_ptr()) +
          static_cast<size_t>(
              o * out_dim_size * inner_size + offset_along_dim * inner_size) *
          elem_size;
      Memcpy(dst, src, nbytes, MemcpyDeviceToDevice);
      offset_along_dim += chunk;
    }
  }

  return out;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kMetax, CatKernelMetax)

} // namespace at::native::flagos
