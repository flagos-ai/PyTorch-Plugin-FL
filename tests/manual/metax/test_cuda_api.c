/*

mxcc -o test_cuda_api test_cuda_api.c \
  -I/opt/maca-3.3.0/tools/cu-bridge/include \
  -L/opt/maca-3.3.0/lib \
  -lruntime_cu

export LD_LIBRARY_PATH=/opt/maca-3.3.0/lib:$LD_LIBRARY_PATH

./test_cuda_api

*/


#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // 1. Allocate Muxi NPU memory (cudaMalloc)
    float* d_data;
    cudaError_t err = cudaMalloc(&d_data, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 2. Create Muxi NPU stream (cudaStreamCreate)
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }

    // 3. Host -> Muxi NPU data copy (cudaMemcpyAsync)
    float h_data[1024] = {1.0f};
    err = cudaMemcpyAsync(d_data, h_data, 1024 * sizeof(float), 
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return -1;
    }

    // 4. Synchronize stream + release resources
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    printf("All operations on Muxi NPU success!\n");
    return 0;
}