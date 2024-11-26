// @file: ./task-1/src/pjlab/bigchip/f16-v2.cu

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cuda_runtime.h>

namespace playground {
    // Implement the matmul function with DType=float16_t and Version=2

    __global__ void matmul(float16_t *A, float16_t *B, float16_t *C, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float16_t value = 0.0;
            for (int k = 0; k < K; ++k) {
                value += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }

    PLAYGROUND_MATMUL_SIG(float16_t, 2, M, N, K, A, B, C) {
        
        float16_t *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float16_t));
        cudaMalloc(&d_B, K * N * sizeof(float16_t));
        cudaMalloc(&d_C, M * N * sizeof(float16_t));

        cudaMemcpy(d_A, A, M * K * sizeof(float16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, K * N * sizeof(float16_t), cudaMemcpyHostToDevice);

        dim3 blockDim(32, 32);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);  // Calculate grid size

        matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

        cudaMemcpy(C, d_C, M * N * sizeof(float16_t), cudaMemcpyDeviceToHost);
    }

    
}