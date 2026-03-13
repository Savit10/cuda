#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdlib>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

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
    CHECK_CUDA(cudaMalloc(&d_A, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // cuBLAS uses column-major, so we compute C = B * A to get row-major result
    // C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
    // For row-major: swap A and B, use transposed dimensions
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,
                             d_A, N,
                             &beta,
                             d_C, N));

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
