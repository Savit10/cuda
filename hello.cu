#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK(call)                                   \
do {                                                  \
    cudaError_t err = call;                           \
    if (err != cudaSuccess) {                         \
        printf("CUDA error %s:%d: %s\n",              \
               __FILE__, __LINE__,                    \
               cudaGetErrorString(err));              \
        exit(1);                                      \
    }                                                 \
} while (0)

__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1500000;

    std::vector<float> h_A(N), h_B(N), h_C(N);

    for (int i = 0; i < N; i++) {
        h_A[i] = float(i + 1);
        h_B[i] = float((i + 1) * 10);
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 512;
    int blocks = (N + blockSize - 1) / blockSize;
    vecAdd<<<blocks, blockSize>>>(d_A, d_B, d_C, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
