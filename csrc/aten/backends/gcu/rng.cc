// Copyright (c) 2026, BAAI. All rights reserved.
//
// Native random-number kernels for Enflame GCU (topsaten).
//
// topsaten's RNG API accepts an explicit {seed, offset} value rather than an
// ATen generator. Generator-less calls consume the same seed/offset stream as
// FlagGems; an explicitly supplied generator remains isolated from that stream.

#include "../../generated/ops.h"
#include "../flagos/python_op_caller.h"
#include "topsaten_common.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Generator.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/exponential.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/poisson.h>
#include <c10/core/Device.h>

#include <limits>
#include <mutex>

namespace at::native::flagos {
namespace {

int64_t next_seed(const ::std::optional<at::Generator>& generator) {
  auto gen = generator.has_value() && generator->defined()
      ? *generator
      : at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(gen.mutex());
  return static_cast<int64_t>(
      at::check_generator<at::CPUGeneratorImpl>(gen)->random64());
}

topsatenGenerator_t make_generator(
    const ::std::optional<at::Generator>& generator,
    int64_t device_index,
    uint64_t increment) {
  if (!generator.has_value() || !generator->defined()) {
    auto state = GetFlagosPhiloxState(device_index, increment);
    return topsatenGenerator_t{state.first, state.second};
  }
  return topsatenGenerator_t{
      static_cast<uint64_t>(next_seed(generator)), 0};
}

topsatenGenerator_t make_generator(
    const ::std::optional<at::Generator>& generator,
    const at::Tensor& output) {
  auto device_index = output.device().index();
  if (device_index < 0) device_index = 0;
  return make_generator(generator, device_index, output.numel());
}

at::Tensor cpu_to_device(const at::Tensor& cpu, const at::Device& device) {
  return cpu.to(device);
}

at::Tensor make_empty(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return at::empty(
      size, at::TensorOptions()
                .dtype(dtype.value_or(at::kFloat))
                .layout(layout.value_or(at::kStrided))
                .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
                .pinned_memory(pin_memory.value_or(false)));
}

bool supported(const at::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0 &&
      gcu::TopsatenSupportsDtype(tensor.scalar_type());
}

at::Tensor randn_cpu(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    const ::std::optional<at::Generator>& generator) {
  auto target = device.value_or(at::Device(at::kPrivateUse1, 0));
  auto cpu = at::empty(
      size, at::TensorOptions()
                .dtype(dtype.value_or(at::kFloat))
                .layout(layout.value_or(at::kStrided))
                .device(at::kCPU)
                .pinned_memory(pin_memory.value_or(false)));
  cpu.normal_(0.0, 1.0, generator);
  return cpu_to_device(cpu, target);
}

at::Tensor randint_cpu(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    const ::std::optional<at::Generator>& generator) {
  auto target = device.value_or(at::Device(at::kPrivateUse1, 0));
  auto cpu = at::empty(
      size, at::TensorOptions()
                .dtype(dtype.value_or(at::kLong))
                .layout(layout.value_or(at::kStrided))
                .device(at::kCPU)
                .pinned_memory(pin_memory.value_or(false)));
  cpu.random_(low, high, generator);
  return cpu_to_device(cpu, target);
}

} // namespace

at::Tensor RandnKernelGcu(
    at::IntArrayRef size, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  if (!supported(out)) return randn_cpu(size, dtype, layout, device, pin_memory, {});
  auto gen = make_generator({}, out);
  gcu::TopsatenSizeWrapper shape(size);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandn, out, t_out.get(), shape.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor RandnGeneratorKernelGcu(
    at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  if (!supported(out)) {
    return randn_cpu(size, dtype, layout, device, pin_memory, generator);
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenSizeWrapper shape(size);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandn, out, t_out.get(), shape.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor RandnLikeGeneratorKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  if (!supported(out)) {
    auto cpu = at::empty_like(self.cpu(), dtype, layout, at::kCPU, pin_memory,
                               memory_format);
    cpu.normal_(0.0, 1.0, generator);
    return cpu_to_device(cpu, self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandnLike, out, t_out.get(), t_self.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false),
                    TOPSATEN_MEMORY_CONTIGUOUS, gen);
  return out;
}

at::Tensor& RandnLikeGeneratorOutKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::MemoryFormat> memory_format, at::Tensor& out) {
  if (!supported(out)) {
    auto cpu = out.cpu();
    cpu.normal_(0.0, 1.0, generator);
    out.copy_(cpu);
    return out;
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandnLike, out, t_out.get(), t_self.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, false,
                    TOPSATEN_MEMORY_CONTIGUOUS, gen);
  return out;
}

at::Tensor RandintLowGeneratorKernelGcu(
    int64_t low, int64_t high, at::IntArrayRef size,
    ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandintGeneratorKernelGcu(
    int64_t high, at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelGcu(0, high, size, generator, dtype, layout,
                                      device, pin_memory);
}

at::Tensor RandintLowGeneratorKernelGcu(
    int64_t low, int64_t high, at::IntArrayRef size,
    ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype.value_or(at::kLong), layout, device, pin_memory);
  if (!supported(out)) {
    return randint_cpu(low, high, size, dtype, layout, device, pin_memory, generator);
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenSizeWrapper shape(size);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandint, out, t_out.get(), low, high, shape.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor RandpermGeneratorKernelGcu(
    int64_t n, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  auto out = make_empty({n}, dtype.value_or(at::kLong), layout, device, pin_memory);
  if (!supported(out)) {
    auto cpu = at::empty({n}, out.options().device(at::kCPU));
    cpu.random_(0, n, generator);
    return cpu_to_device(cpu, out.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandperm, out, t_out.get(), n,
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor& RandomInplaceKernelGcu(
    at::Tensor& self, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.random_(generator);
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandom, self, t_self.get(), gen);
  return self;
}

at::Tensor& RandomInplaceToKernelGcu(
    at::Tensor& self, int64_t to, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.random_(to, generator);
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandom, self, t_self.get(), to, gen);
  return self;
}

at::Tensor& BernoulliInplaceFloatKernelGcu(
    at::Tensor& self, double p, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.bernoulli_(p, generator);
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenBernoulli, self, t_self.get(), p, gen);
  return self;
}

at::Tensor BernoulliKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(self);
  if (!supported(out)) return at::bernoulli(self.cpu(), generator).to(self.device());
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenBernoulli, out, t_out.get(), t_self.get(), gen);
  return out;
}

at::Tensor ExponentialKernelGcu(
    const at::Tensor& self, double lambda, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(self);
  if (!supported(out)) return at::exponential(self.cpu(), lambda, generator).to(self.device());
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenExponential, out, t_out.get(), lambda, gen);
  return out;
}

at::Tensor& ExponentialInplaceKernelGcu(
    at::Tensor& self, double lambda, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.exponential_(lambda, generator);
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenExponential, self, t_self.get(), lambda, gen);
  return self;
}

at::Tensor MultinomialKernelGcu(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    ::std::optional<at::Generator> generator) {
  auto out = at::empty({self.size(0), num_samples},
                       self.options().dtype(at::kLong));
  if (!supported(self) || !gcu::TopsatenSupportsDtype(out.scalar_type())) {
    return at::multinomial(self.cpu(), num_samples, replacement, generator)
        .to(self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenMultinomial, out, t_out.get(), t_self.get(),
                    num_samples, replacement, gen);
  return out;
}

at::Tensor PoissonKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(self);
  if (!supported(self)) return at::poisson(self.cpu(), generator).to(self.device());
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenPoisson, out, t_out.get(), t_self.get(), gen);
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandnFn, randn_dispatcher, Backend::kGcu, RandnKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnGeneratorFn, randn_generator_dispatcher, Backend::kGcu, RandnGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorFn, randn_like_generator_dispatcher, Backend::kGcu, RandnLikeGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorOutFn, randn_like_generator_out_dispatcher, Backend::kGcu, RandnLikeGeneratorOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintGeneratorFn, randint_generator_dispatcher, Backend::kGcu, RandintGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLowGeneratorFn, randint_low_generator_dispatcher, Backend::kGcu, RandintLowGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandpermGeneratorFn, randperm_generator_dispatcher, Backend::kGcu, RandpermGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFn, random_inplace_dispatcher, Backend::kGcu, RandomInplaceKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceToFn, random_inplace_to_dispatcher, Backend::kGcu, RandomInplaceToKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(BernoulliFn, bernoulli_dispatcher, Backend::kGcu, BernoulliKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceFloatFn, bernoulli_inplace_float_dispatcher, Backend::kGcu, BernoulliInplaceFloatKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(ExponentialFn, exponential_dispatcher, Backend::kGcu, ExponentialKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(ExponentialInplaceFn, exponential_inplace_dispatcher, Backend::kGcu, ExponentialInplaceKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(MultinomialFn, multinomial_dispatcher, Backend::kGcu, MultinomialKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(PoissonFn, poisson_dispatcher, Backend::kGcu, PoissonKernelGcu)

} // namespace at::native::flagos
