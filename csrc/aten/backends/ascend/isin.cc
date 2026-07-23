// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/isin.h>

namespace at::native::flagos {

// isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
//
// CANN has no aclnnIsIn kernel, and composing it on-device
// ((elements.unsqueeze(-1) == test_elements).any(-1)) would need several more
// registered ops. In practice transformers only calls this on tiny token-id
// tensors (eos/pad ids) inside generate(), so computing on CPU and copying the
// bool result back is both correct and negligibly cheap. This is a pragmatic
// bespoke kernel, not a hot path.
at::Tensor IsinTensorTensorKernelAscend(const at::Tensor& elements,
                                        const at::Tensor& test_elements,
                                        bool assume_unique, bool invert) {
  auto elements_cpu = elements.cpu();
  auto test_cpu = test_elements.cpu();
  auto out_cpu = at::isin(elements_cpu, test_cpu, assume_unique, invert);
  return out_cpu.to(elements.device());
}

REGISTER_IMPL_TO_DISPATCHER(IsinTensorTensorFn, isin_tensor_tensor_dispatcher, Backend::kAscend, IsinTensorTensorKernelAscend)

} // namespace at::native::flagos
