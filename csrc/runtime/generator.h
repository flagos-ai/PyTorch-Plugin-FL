// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.h
// with OpenRegGeneratorImpl renamed to GeneratorImpl and getDefaultOpenRegGenerator renamed to getDefaultGenerator.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include <c10/core/Device.h>

#include "functions.h"

namespace c10::flagos {

class GeneratorImpl : public at::CPUGeneratorImpl {
 public:
  GeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~GeneratorImpl() override = default;
};

const at::Generator& GetDefaultGenerator(
    c10::DeviceIndex device_index = -1);

} // namespace c10::flagos
