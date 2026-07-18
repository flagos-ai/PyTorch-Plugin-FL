// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include "dispatcher.h"

namespace at::native::flagos {

using IndexTensorFn = at::Tensor (*)(const at::Tensor&, const c10::List<::std::optional<at::Tensor>>&);
DECLARE_DISPATCHER(IndexTensorFn, index_tensor_dispatcher)

} // namespace at::native::flagos
