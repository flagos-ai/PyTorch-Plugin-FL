// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../embedding.h"

#include <ATen/core/ivalue.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingKernelPython(const at::Tensor& weight, const at::Tensor& indices,
                                 int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  std::vector<c10::IValue> args;
  args.emplace_back(weight);
  args.emplace_back(indices);
  args.emplace_back(padding_idx);
  args.emplace_back(scale_grad_by_freq);
  args.emplace_back(sparse);
  return CallPythonOp_Generic("embedding", args);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(EmbeddingFn, embedding_stub, FlagosDevice::kFlagOsPython, EmbeddingKernelPython)

} // namespace at::native::flagos
