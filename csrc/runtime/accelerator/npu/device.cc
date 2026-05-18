#include <include/flagos.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstdio>

namespace {

thread_local int gCurrentDevice = 0;

bool ensureAclInit() {
  static bool initialized = []() {
    aclError err = aclInit(nullptr);
    if (err != ACL_SUCCESS && err != ACL_ERROR_REPEAT_INITIALIZE) {
      fprintf(stderr, "[flagos-npu] aclInit failed: %d\n", static_cast<int>(err));
      return false;
    }
    return true;
  }();
  return initialized;
}

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  ensureAclInit();

  uint32_t npu_count = 0;
  aclError err = aclrtGetDeviceCount(&npu_count);
  if (err != ACL_SUCCESS) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = static_cast<int>(npu_count);
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  static bool first_call = []() {
    ensureAclInit();
    aclrtSetDevice(0);
    return true;
  }();
  (void)first_call;

  *device = gCurrentDevice;
  return Success;
}

Error_t SetDevice(int device) {
  ensureAclInit();

  int count = 0;
  GetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  aclError err = aclrtSetDevice(device);
  if (err != ACL_SUCCESS) {
    return ErrorUnknown;
  }

  gCurrentDevice = device;
  return Success;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  *leastPriority = 0;
  *greatestPriority = 0;
  return Success;
}

Error_t DeviceSynchronize(void) {
  aclError err = aclrtSynchronizeDevice();
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}
