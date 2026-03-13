#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>

using namespace std;
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

__global__ void matmul(float *a, float *b, float *c, int N) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (column < N && row < N) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * N + column];
    }
    c[row * N + column] = sum;
  }
}

int main(int argc, char *argv[]) {
    int N = 16;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    const int size = N * N;

    std::vector<float> h_A(size), h_B(size), h_C(size);

    for (int i = 0; i < size; i++) {
        h_A[i] = float(i + 1);
        h_B[i] = float(i + 1);
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, size * sizeof(float)));
    CHECK(cudaMalloc(&d_B, size * sizeof(float)));
    CHECK(cudaMalloc(&d_C, size * sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (N + 31) / 32);
    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
