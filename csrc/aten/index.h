#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include "dispatcher.h"

namespace at::native::flagos {

using IndexTensorFn = at::Tensor (*)(const at::Tensor&, const c10::List<::std::optional<at::Tensor>>&);
DECLARE_DISPATCHER(IndexTensorFn, index_tensor_dispatcher)

} // namespace at::native::flagos
