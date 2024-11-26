// @file: ./task-1/src/pjlab/bigchip/f16-v2.cu

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cuda_runtime.h>

namespace playground
{
// Implement the matmul function with DType=float16_t and Version=2

const int BLOCK = 32;
const int STRIDE = 4;

__global__ void matmul_v3(float16_t* a, float16_t* b, float16_t* c, int m, int n, int k)
{
    constexpr int STEP = BLOCK * STRIDE;
    const int tx = threadIdx.x * STRIDE;
    const int ty = threadIdx.y * STRIDE;
    const int bx = blockIdx.x * STEP;
    const int by = blockIdx.y * STEP;

    float16_t* begin_a = a + by * k;
    float16_t* begin_b = b + bx;
    float16_t* end_a = begin_a + k;

    float sum[STRIDE][STRIDE] = {0.f};
    for (float16_t *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
         a_ptr += STEP, b_ptr += STEP * n) {
        __shared__ __align__(16 * 1024) float ashare[STEP][STEP];
        __shared__ __align__(16 * 1024) float bshare[STEP][STEP];

        for (int i = 0; i < STRIDE; ++i) {
            for (int j = 0; j < STRIDE; ++j) {
                ashare[ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];
                bshare[ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];
            }
        }
        __syncthreads();

        for (int i = 0; i < STRIDE; ++i) {
            for (int j = 0; j < STRIDE; ++j) {
                for (int kk = 0; kk < STEP; ++kk) {
                    sum[i][j] += ashare[ty + i][kk] * bshare[kk][tx + j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
        for (int j = 0; j < STRIDE; ++j) {
            c[(by + ty + i) * n + bx + tx + j] = sum[i][j];
        }
    }
}

PLAYGROUND_MATMUL_SIG(float16_t, 3, M, N, K, A, B, C)
{

    float16_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float16_t));
    cudaMalloc(&d_B, K * N * sizeof(float16_t));
    cudaMalloc(&d_C, M * N * sizeof(float16_t));

    cudaMemcpy(d_A, A, M * K * sizeof(float16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float16_t), cudaMemcpyHostToDevice);

    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK / STRIDE, (N + BLOCK - 1) / BLOCK / STRIDE);

    matmul_v3<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, M * N * sizeof(float16_t), cudaMemcpyDeviceToHost);
}

}

