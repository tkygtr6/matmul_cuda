#include "mymulmat.h"

#include <cstdio>
#include <cuda_runtime.h>


void printError(cudaError_t e, int l) {
    if (e != cudaSuccess) {
        printf("error: %s (code %d), line(%d)\n", cudaGetErrorString(e), e, l);
        exit(EXIT_FAILURE);
    }
}

#define CheckError(f) printError((f), __LINE__)

#define BX 32
#define BY 8
#define STEP 32
#define UNX 8
#define UNY 16
#define G_A_INDEX(b_x, b_y, t_x, t_y) ((n) * ((BY) * (b_y) + (t_y)) + ((STEP) * (b_x) + (t_x)))
#define G_B_INDEX(b_x, b_y, t_x, t_y) ((n) * ((STEP) * (b_y) + (t_y)) + ((BX) * (b_x) + (t_x)))
#define G_C_INDEX(b_x, b_y, t_x, t_y) ((n) * ((BY) * (b_y) + (t_y)) + ((BX) * (b_x) + (t_x)))
#define S_INDEX(t_x, t_y, w_x) ((t_y) * (w_x) + (t_x))

__global__
void kernel(int n, int m, int k, melem_t *A, melem_t *B, melem_t *C) {
    __shared__ melem_t A_[STEP * (BY * UNY)];
    __shared__ melem_t B_[(BX * UNX) * STEP];
    melem_t C_[UNX * UNY];
    for(int i = 0; i < UNX; i++){
        for(int j = 0; j < UNY; j++){
            C_[S_INDEX(i, j, UNX)] = 0;
        }
    }
    for(int s = 0; s < n / STEP; s++){
        for(int t = 0; t < STEP/BX; t++){
            for(int i = 0; i < UNY; i++){
                A_[S_INDEX(threadIdx.x + BX * t, threadIdx.y + BY * i, STEP)] = A[G_A_INDEX(s, blockIdx.y * UNY + i, threadIdx.x + BX * t, threadIdx.y)];
            }
        }
        for(int t = 0; t < STEP/BY; t++){
            for(int i = 0; i < UNX; i++){
                B_[S_INDEX(threadIdx.x + BX * i, threadIdx.y + BY * t, BX * UNX)] = B[G_B_INDEX(blockIdx.x * UNX + i, s, threadIdx.x, threadIdx.y + BY * t)];
            }
        }
        __syncthreads();

        for(int k = 0; k < STEP; k++){
            for(int i = 0; i < UNX; i++){
                for(int j = 0; j < UNY; j++){
                    C_[S_INDEX(i, j, UNX)] += A_[S_INDEX(k, threadIdx.y + BY * j, STEP)] * B_[S_INDEX(threadIdx.x + BX * i, k, BX * UNX)];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < UNX; i++){
        for(int j = 0; j < UNY; j++){
            C[G_C_INDEX(blockIdx.x * UNX + i, blockIdx.y * UNY + j, threadIdx.x, threadIdx.y)] = C_[S_INDEX(i, j, UNX)];
        }
    }
}

uint64_t cudaGemm(int n, int m, int k, melem_t *A, melem_t *B, melem_t *C) {
    // device initialize
    int device = 0;
    cudaSetDevice(device);

    // device malloc
    melem_t *devA, *devB, *devC;
    size_t sizeA = size_t(n)*k*sizeof(melem_t);
    size_t sizeB = size_t(k)*m*sizeof(melem_t);
    size_t sizeC = size_t(n)*m*sizeof(melem_t);

    CheckError(cudaMalloc((void**) &devA, sizeA));
    CheckError(cudaMalloc((void**) &devB, sizeB));
    CheckError(cudaMalloc((void**) &devC, sizeC));

    // data load
    CheckError(cudaMemcpy(devA, A, sizeA, cudaMemcpyHostToDevice));
    CheckError(cudaMemcpy(devB, B, sizeB, cudaMemcpyHostToDevice));
    CheckError(cudaMemcpy(devC, C, sizeC, cudaMemcpyHostToDevice));

    // gemm start
    cudaEvent_t start, stop;
    CheckError(cudaEventCreate(&start));
    CheckError(cudaEventCreate(&stop));
    cudaDeviceSynchronize();

    dim3 grid(n / (BX * UNX), n / (BY * UNY));
    dim3 block(BX, BY);
    printf("Shared memory: %d\n", (STEP * BY * UNY + BX * UNX * STEP) * 4);

    // time measuring
    CheckError(cudaEventRecord(start, NULL));
    kernel <<<grid, block>>> (n, m, k, devA, devB, devC);
    CheckError(cudaEventRecord(stop, NULL));

    // gemm end
    CheckError(cudaEventSynchronize(stop));
    float msec = 0.0f;
    CheckError(cudaEventElapsedTime(&msec, start, stop));

    // data store
    CheckError(cudaMemcpy(C, devC, sizeC, cudaMemcpyDeviceToHost));

    // device free
    CheckError(cudaFree(devA));
    CheckError(cudaFree(devB));
    CheckError(cudaFree(devC));

    return (uint64_t)(msec * 1000.0f);
}
