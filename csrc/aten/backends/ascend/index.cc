// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../index.h"
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor IndexTensorKernelAscend(const at::Tensor& self,
                                   const c10::List<::std::optional<at::Tensor>>& indices) {
  namespace ascend = at::native::flagos::ascend;

  std::vector<const aclTensor*> acl_indices_ptrs;
  std::vector<ascend::AclTensorWrapper> acl_indices_storage;
  acl_indices_storage.reserve(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    const auto& idx = indices.get(i);
    if (idx.has_value()) {
      acl_indices_storage.emplace_back(idx.value());
      acl_indices_ptrs.push_back(acl_indices_storage.back().get());
    }
  }

  ascend::AclTensorListWrapper acl_indices(acl_indices_ptrs);
  ascend::AclTensorWrapper acl_self(self);

  auto out_sizes = self.sizes().vec();
  if (!acl_indices_ptrs.empty() && indices.get(0).has_value()) {
    auto idx_shape = indices.get(0).value().sizes().vec();
    out_sizes.erase(out_sizes.begin(), out_sizes.begin() + static_cast<int64_t>(acl_indices_ptrs.size()));
    out_sizes.insert(out_sizes.begin(), idx_shape.begin(), idx_shape.end());
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, self.options());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnIndex, acl_self.get(), acl_indices.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(IndexTensorFn, index_tensor_stub, FlagosDevice::kAscend, IndexTensorKernelAscend)

} // namespace at::native::flagos
