// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../embedding_dense_backward.h"

#include <ATen/core/ivalue.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingDenseBackwardKernelPython(const at::Tensor& grad_output, const at::Tensor& indices,
                                              int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  std::vector<c10::IValue> args;
  args.emplace_back(grad_output);
  args.emplace_back(indices);
  args.emplace_back(num_weights);
  args.emplace_back(padding_idx);
  args.emplace_back(scale_grad_by_freq);
  return CallPythonOp_Generic("embedding_dense_backward", args);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(EmbeddingDenseBackwardFn, embedding_dense_backward_dispatcher, Backend::kFlagOsPython, EmbeddingDenseBackwardKernelPython)

} // namespace at::native::flagos
