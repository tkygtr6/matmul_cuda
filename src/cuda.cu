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
#define BY 32
#define G_INDEX(b_x, b_y, t_x, t_y) (n * (BY * b_y + t_y) + (BX * b_x + t_x))
#define S_INDEX(t_x, t_y) (blockDim.x * t_y + t_x)

__global__
void kernel(int n, int m, int k, melem_t *A, melem_t *B, melem_t *C) {
    __shared__ melem_t A_[BX * BY];
    __shared__ melem_t B_[BX * BY];
    __shared__ melem_t C_[BX * BY];

    C_[S_INDEX(threadIdx.x, threadIdx.y)] = 0;
    for(int t = 0; t < n / BX; t++){
        A_[S_INDEX(threadIdx.x, threadIdx.y)] = A[G_INDEX(t, blockIdx.y, threadIdx.x, threadIdx.y)];
        B_[S_INDEX(threadIdx.x, threadIdx.y)] = B[G_INDEX(blockIdx.x, t, threadIdx.x, threadIdx.y)];
        __syncthreads();

        for(int s = 0; s < BX; s++){
            C_[S_INDEX(threadIdx.x, threadIdx.y)] += A_[S_INDEX(s, threadIdx.y)] * B_[S_INDEX(threadIdx.x, s)];
        }
        __syncthreads();
    }

    C[G_INDEX(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y)] = C_[S_INDEX(threadIdx.x, threadIdx.y)];
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

    dim3 grid(n / BX, n / BY);
    dim3 block(BX, BY);

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
