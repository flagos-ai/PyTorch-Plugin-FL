// Copyright 2026 FlagOS Contributors. Apache-2.0.
#include <torch/csrc/profiler/stubs/base.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <functional>
#include <memory>

#include <include/flagos.h>  // flagos C ABI: Event_t, EventCreateWithFlags, ...

namespace c10::flagos {
namespace {

using torch::profiler::impl::ProfilerStubs;
using torch::profiler::impl::ProfilerVoidEventStub;

struct FlagosProfilerStubs : public ProfilerStubs {
  void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override {
    if (device) {
      int d = 0;
      ::GetDevice(&d);
      *device = static_cast<c10::DeviceIndex>(d);
    }
    if (cpu_ns) {
      *cpu_ns = c10::getTime();
    }
    Event_t ev = nullptr;
    ::EventCreateWithFlags(&ev, EventEnableTiming);
    ::EventRecord(ev, nullptr);  // 记在当前(默认)流；多流由 guard 路径覆盖
    *event = std::shared_ptr<void>(ev, [](void* p) {
      if (p) ::EventDestroy((Event_t)p);
    });
  }

  float elapsed(const ProfilerVoidEventStub* event,
                const ProfilerVoidEventStub* event2) const override {
    ::EventSynchronize((Event_t)event2->get());
    float ms = 0.0f;
    ::EventElapsedTime(&ms, (Event_t)event->get(), (Event_t)event2->get());
    return ms * 1000.0f;  // µs
  }

  void mark(const char*) const override {}       // Stage A: no-op (NVTX will come later)
  void rangePush(const char*) const override {}
  void rangePop() const override {}
  bool enabled() const override { return true; }

  void onEachDevice(std::function<void(int)> op) const override {
    int count = 0;
    ::GetDeviceCount(&count);
    for (int i = 0; i < count; ++i) op(i);
  }

  void synchronize() const override { ::DeviceSynchronize(); }

  ~FlagosProfilerStubs() override = default;
};

struct RegisterFlagosStubs {
  RegisterFlagosStubs() {
    static FlagosProfilerStubs stubs;
    torch::profiler::impl::registerPrivateUse1Methods(&stubs);
  }
};
static RegisterFlagosStubs g_register_flagos_stubs;

}  // namespace
}  // namespace c10::flagos
