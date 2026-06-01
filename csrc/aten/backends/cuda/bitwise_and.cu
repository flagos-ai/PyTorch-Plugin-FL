// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bitwise_and.h"

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>

#include "native/Loops.cuh"

namespace at::native::flagos {

namespace {

at::Tensor BitwiseAndKernelCuda(const at::Tensor& self, const at::Tensor& other) {
  auto result_type = at::result_type(self, other);
  auto self_cast = self.to(result_type);
  auto other_cast = other.to(result_type);

  auto result_shape = at::infer_size(self_cast.sizes(), other_cast.sizes());
  at::Tensor output = at::empty(result_shape, self_cast.options());
  auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(self_cast)
    .add_input(other_cast)
    .build();
  AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, result_type, "bitwise_and_cuda", [&]() {
    at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a & b;
    });
  });
  return output;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(BitwiseAndTensorFn, bitwise_and_tensor_dispatcher, Backend::kCuda, BitwiseAndKernelCuda)

} // namespace at::native::flagos
