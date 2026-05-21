// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mean.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor MeanDimKernelAscend(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  namespace ascend = at::native::flagos::ascend;

  auto out_dtype = dtype.value_or(self.scalar_type());

  // Compute output shape
  auto in_sizes = self.sizes().vec();
  std::vector<int64_t> out_sizes;
  std::vector<int64_t> dims;
  if (dim.has_value()) {
    dims = dim.value().vec();
    for (auto& d : dims) {
      if (d < 0) d += self.dim();
    }
  } else {
    for (int64_t i = 0; i < self.dim(); ++i) {
      dims.push_back(i);
    }
  }

  std::set<int64_t> dim_set(dims.begin(), dims.end());
  for (int64_t i = 0; i < static_cast<int64_t>(in_sizes.size()); ++i) {
    if (dim_set.count(i)) {
      if (keepdim) out_sizes.push_back(1);
    } else {
      out_sizes.push_back(in_sizes[i]);
    }
  }
  if (out_sizes.empty()) out_sizes.push_back(1);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  auto acl_dims = ascend::AclIntArrayWrapper(dims);

  EXEC_ASCEND_CMD(aclnnMean, acl_self.get(), acl_dims.get(), keepdim, out_dtype, acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(MeanDimFn, mean_dim_stub, FlagosDevice::kAscend, MeanDimKernelAscend)

} // namespace at::native::flagos
