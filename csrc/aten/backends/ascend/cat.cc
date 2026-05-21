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

  auto first = materialized[0].get();
  auto ndim = first.dim();
  if (dim < 0) dim += ndim;

  // Compute output shape
  std::vector<int64_t> out_sizes(first.sizes().begin(), first.sizes().end());
  for (size_t i = 1; i < materialized.size(); ++i) {
    out_sizes[dim] += materialized[i].get().size(dim);
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, first.options());

  // Build aclTensorList
  std::vector<const aclTensor*> acl_tensors;
  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(materialized.size());
  for (const auto& t : materialized) {
    wrappers.emplace_back(t.get());
  }
  for (auto& w : wrappers) {
    acl_tensors.push_back(w.get());
  }

  aclTensorList* tensor_list = aclCreateTensorList(
      acl_tensors.data(), acl_tensors.size());

  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnCat, tensor_list, dim, acl_out.get());

  aclDestroyTensorList(tensor_list);
  return out;
}

FLAGOS_REGISTER_DISPATCH(CatFn, cat_stub, FlagosDevice::kAscend, CatKernelAscend)

} // namespace at::native::flagos
