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

__global__
void kernel(int n, int m, int k, melem_t *A, melem_t *B, melem_t *C) {
    for (int i = 0; i < n; i++) {
        for (int j= 0; j < m; j++) {
            for (int l = 0; l <k; l++) {
                C[i*m+j] += A[i*k+l] * B[l*m+j];
            }
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

    // time measuring
    CheckError(cudaEventRecord(start, NULL));
    kernel <<<1, 1>>> (n, m, k, devA, devB, devC);
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
