// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor EmbeddingKernelAscend(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  namespace ascend = at::native::flagos::ascend;

  auto out_sizes = indices.sizes().vec();
  out_sizes.push_back(weight.size(1));
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, weight.options());

  ascend::AclTensorWrapper acl_weight(weight);
  ascend::AclTensorWrapper acl_indices(indices);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnEmbedding, acl_weight.get(), acl_indices.get(), acl_out.get());
  return out;
}

FLAGOS_REGISTER_DISPATCH(EmbeddingFn, embedding_stub, FlagosDevice::kAscend, EmbeddingKernelAscend)

} // namespace at::native::flagos
