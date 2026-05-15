#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using IndexTensorFn = at::Tensor (*)(const at::Tensor&, const c10::List<::std::optional<at::Tensor>>&);
FLAGOS_DECLARE_DISPATCH(IndexTensorFn, index_tensor_stub)

} // namespace at::native::flagos
