// @file: ./task-1/src/pjlab/bigchip/f16-v2.cu

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cstdio>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

namespace playground
{

__global__ void matmul_f32_v3(const float32_t* a, const float32_t* b, float32_t* c, int M,
                              int N, int K)
{
    // ::printf("%g\n", a[0]);
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // alignment to 16k, which is helpful, 5.42 -> 8.01 TFLOPS
    __shared__ __align__(16 * 1024) float s_a[BM][BK];
    __shared__ __align__(16 * 1024) float s_b[BK][BN];

    // __shared__ float s_a[BM][BK];
    // __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {{0.0}};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4_CONST(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4_CONST(b[load_b_gmem_addr]);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }

}

    PLAYGROUND_MATMUL_SIG(float32_t, 3, M, N, K, A, B, C)
    {
        const dim3 BlockSize(16, 16);
        const dim3 GridSize(32, 32);

        matmul_f32_v3<<<GridSize, BlockSize>>>(A, B, C, M, N, K);
    }

}

