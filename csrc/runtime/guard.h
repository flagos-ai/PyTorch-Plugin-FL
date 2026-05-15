// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <include/flagos.h>

namespace c10::flagos {

struct GuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  GuardImpl() = default;
  explicit GuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    auto old_device_index = exchangeDeviceIndex(d.index());
    return c10::Device(c10::DeviceType::PrivateUse1, old_device_index);
  }

  c10::DeviceIndex exchangeDeviceIndex(c10::DeviceIndex device_index) const {
    int prev_device = -1;
    ::GetDevice(&prev_device);
    if (prev_device != device_index) {
      ::SetDevice(device_index);
    }
    return static_cast<c10::DeviceIndex>(prev_device);
  }

  c10::Device getDevice() const override {
    int device = -1;
    ::GetDevice(&device);
    return c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device));
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    ::SetDevice(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    ::SetDevice(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    // Return stream with ID 0 (not DEFAULT which is -1)
    // The autograd engine expects valid stream IDs
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::UNSAFE, s.device(), 0);
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }

  bool queryStream(const c10::Stream& stream) const override {
    // Synchronize CUDA to ensure all operations are complete
    ::DeviceSynchronize();
    return true;
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    ::DeviceSynchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (event) {
      ::EventSynchronize((Event_t)event);
    }
  }

  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    // No-op: flagos uses CUDA memory which is already tracked
  }

  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    float ms = 0.0f;
    if (event1 && event2) {
      ::EventElapsedTime(&ms, (Event_t)event1, (Event_t)event2);
    }
    return static_cast<double>(ms);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    int count = 0;
    ::GetDeviceCount(&count);
    return static_cast<c10::DeviceIndex>(count);
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    ::EventCreate((Event_t*)event);
    ::EventRecord(*(Event_t*)event, nullptr);
  }

  void block(void* event, const c10::Stream& stream) const override {
    ::StreamWaitEvent(nullptr, (Event_t)event, 0);
  }

  bool queryEvent(void* event) const override {
    return ::EventQuery((Event_t)event) == Success;
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    ::EventDestroy((Event_t)event);
  }
};

} // namespace c10::flagos
