// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding_dense_backward.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor EmbeddingDenseBackwardKernelAscend(const at::Tensor& grad_output,
                                              const at::Tensor& indices,
                                              int64_t num_weights,
                                              int64_t padding_idx,
                                              bool scale_grad_by_freq) {
  namespace ascend = at::native::flagos::ascend;
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      {num_weights, grad_output.size(-1)}, grad_output.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_indices(indices);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);

  EXEC_ASCEND_CMD(aclnnEmbeddingDenseBackward, acl_grad_output.get(),
                  acl_indices.get(), num_weights, padding_idx,
                  scale_grad_by_freq, acl_grad_weight.get());
  return grad_weight;
}

FLAGOS_REGISTER_DISPATCH(EmbeddingDenseBackwardFn, embedding_dense_backward_stub, FlagosDevice::kAscend, EmbeddingDenseBackwardKernelAscend)

} // namespace at::native::flagos
