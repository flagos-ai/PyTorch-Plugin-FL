// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cat.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor CatKernelAscend(const at::ITensorListRef& tensors, int64_t dim) {
  namespace ascend = at::native::flagos::ascend;

  auto materialized = tensors.materialize();
  TORCH_CHECK(!materialized.empty(), "cat: expected a non-empty list of tensors");

  // Filter out empty tensors (numel == 0) to avoid dimension mismatches
  std::vector<at::Tensor> valid_tensors;
  for (const auto& t : materialized) {
    if (t.get().numel() > 0) {
      valid_tensors.push_back(t.get());
    }
  }

  // If all tensors are empty, return the first one
  if (valid_tensors.empty()) {
    return materialized[0].get().clone();
  }

  // If only one non-empty tensor, return it directly
  if (valid_tensors.size() == 1) {
    return valid_tensors[0].clone();
  }

  auto& first = valid_tensors[0];
  auto ndim = first.dim();
  if (dim < 0) dim += ndim;

  // Compute output shape
  std::vector<int64_t> out_sizes(first.sizes().begin(), first.sizes().end());
  for (size_t i = 1; i < valid_tensors.size(); ++i) {
    out_sizes[dim] += valid_tensors[i].size(dim);
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, first.options());

  // Build aclTensorList
  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(valid_tensors.size());
  for (auto& t : valid_tensors) {
    wrappers.emplace_back(t);
  }

  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(valid_tensors.size());
  for (auto& w : wrappers) {
    acl_tensors.push_back(w.get());
  }

  aclTensorList* tensor_list = aclCreateTensorList(
      acl_tensors.data(), acl_tensors.size());

  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnCat, tensor_list, dim, acl_out.get());

  // Do not call aclDestroyTensorList — it may destroy the internal aclTensor
  // objects which are still owned by the AclTensorWrapper RAII objects.
  (void)tensor_list;
  return out;
}

FLAGOS_REGISTER_DISPATCH(CatFn, cat_stub, FlagosDevice::kAscend, CatKernelAscend)

} // namespace at::native::flagos
