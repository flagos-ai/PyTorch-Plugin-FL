// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../../mm.h"

#include <ATen/core/Tensor.h>
#include "csrc/aten/npu_adapter/op_preparation.h"
#include "csrc/aten/npu_adapter/op_api_common.h"

namespace at::native::flagos {

void MmKernelNpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  namespace npu = at::native::flagos::npu;
  int8_t cube_math_type = npu::OpPreparation::get_cube_math_type(false);
  EXEC_NPU_CMD(aclnnMm, self, mat2, out, cube_math_type);
}

FLAGOS_REGISTER_DISPATCH(MmFn, mm_stub, FlagosDevice::kNpu, MmKernelNpu)

} // namespace at::native::flagos
