// @file: ./task-1/src/pjlab/bigchip/f32-v2.cu

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cuda_runtime.h>

namespace playground
{
// Implement the matmul function with DType=float32_t and Version=2

const int BLOCK = 32;

__global__ void matmul_f32_v2(const float32_t* A, const float32_t* B, float32_t* C, int M, int N,
                       int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float32_t value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

PLAYGROUND_MATMUL_SIG(float32_t, 2, M, N, K, A, B, C)
{
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    matmul_f32_v2<<<grid, block>>>(A, B, C, M, N, K);
}

}

