#include <include/flagos.h>
#include <cuda_runtime.h>

namespace {

thread_local int gCurrentDevice = 0;

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }
  int dev_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&dev_count);
  if (err != cudaSuccess) {
    *count = 0;
    return ErrorUnknown;
  }
  *count = dev_count;
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }
  *device = gCurrentDevice;
  return Success;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);
  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }
  const cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return ErrorUnknown;
  }
  gCurrentDevice = device;
  return Success;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  const cudaError_t err =
      cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t DeviceSynchronize(void) {
  const cudaError_t err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}
