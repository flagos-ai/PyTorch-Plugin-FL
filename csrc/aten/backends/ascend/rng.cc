// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

// Draw a fresh 64-bit seed from the default CPU generator so successive RNG
// calls decorrelate. offset is left at 0 (aclnn advances its own state).
int64_t next_seed() {
  auto gen = at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(gen.mutex());
  return static_cast<int64_t>(
      at::check_generator<at::CPUGeneratorImpl>(gen)->random64());
}

at::Tensor make_empty(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype,
                      ::std::optional<at::Layout> layout,
                      ::std::optional<at::Device> device,
                      ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
      .dtype(dtype.value_or(at::kFloat))
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
      .pinned_memory(pin_memory.value_or(false));
  return at::empty(size, options);
}

} // namespace

// randn(int[] size, *, ScalarType?, Layout?, Device?, bool? pin_memory)
//   -> Tensor of N(0, 1) samples. aclnnInplaceNormal(selfRef, mean, std, seed, offset).
at::Tensor RandnKernelAscend(at::IntArrayRef size,
                             ::std::optional<at::ScalarType> dtype,
                             ::std::optional<at::Layout> layout,
                             ::std::optional<at::Device> device,
                             ::std::optional<bool> pin_memory) {
  namespace ascend = at::native::flagos::ascend;
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  ascend::AclTensorWrapper acl_out(out);
  EXEC_ASCEND_CMD(aclnnInplaceNormal, const_cast<aclTensor*>(acl_out.get()),
                  0.0f, 1.0f, next_seed(), static_cast<int64_t>(0));
  return out;
}

// rand(int[] size, ...) -> Tensor of U[0, 1) samples.
// aclnnInplaceUniform(selfRef, from, to, seed, offset).
at::Tensor RandKernelAscend(at::IntArrayRef size,
                            ::std::optional<at::ScalarType> dtype,
                            ::std::optional<at::Layout> layout,
                            ::std::optional<at::Device> device,
                            ::std::optional<bool> pin_memory) {
  namespace ascend = at::native::flagos::ascend;
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  ascend::AclTensorWrapper acl_out(out);
  EXEC_ASCEND_CMD(aclnnInplaceUniform, const_cast<aclTensor*>(acl_out.get()),
                  0.0, 1.0, static_cast<uint64_t>(next_seed()),
                  static_cast<uint64_t>(0));
  return out;
}

// randint.low(int low, int high, int[] size, ...) -> Tensor of ints in [low, high).
// aclnnInplaceRandom(selfRef, from, to, seed, offset).
at::Tensor RandintLowKernelAscend(int64_t low, int64_t high, at::IntArrayRef size,
                                  ::std::optional<at::ScalarType> dtype,
                                  ::std::optional<at::Layout> layout,
                                  ::std::optional<at::Device> device,
                                  ::std::optional<bool> pin_memory) {
  namespace ascend = at::native::flagos::ascend;
  // randint defaults to int64 output when no dtype is given.
  auto out = make_empty(size, dtype.value_or(at::kLong), layout, device, pin_memory);
  ascend::AclTensorWrapper acl_out(out);
  EXEC_ASCEND_CMD(aclnnInplaceRandom, const_cast<aclTensor*>(acl_out.get()),
                  low, high, next_seed(), static_cast<int64_t>(0));
  return out;
}

// randint(int high, int[] size, ...) -> ints in [0, high). Delegates to the
// low overload with low=0.
at::Tensor RandintKernelAscend(int64_t high, at::IntArrayRef size,
                               ::std::optional<at::ScalarType> dtype,
                               ::std::optional<at::Layout> layout,
                               ::std::optional<at::Device> device,
                               ::std::optional<bool> pin_memory) {
  return RandintLowKernelAscend(0, high, size, dtype, layout, device, pin_memory);
}

REGISTER_IMPL_TO_DISPATCHER(RandnFn, randn_dispatcher, Backend::kAscend, RandnKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandFn, rand_dispatcher, Backend::kAscend, RandKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintFn, randint_dispatcher, Backend::kAscend, RandintKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintLowFn, randint_low_dispatcher, Backend::kAscend, RandintLowKernelAscend)

} // namespace at::native::flagos
