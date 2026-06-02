// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.cpp
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "device_allocator.h"

namespace c10::flagos {

static DeviceAllocator global_device_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_device_alloc);

} // namespace c10::flagos
