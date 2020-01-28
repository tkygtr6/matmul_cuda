#pragma once

#include <imulmat.h>
#include <cstdint>

class MyMulMat : public IMulMat
{
public:
    MyMulMat();
    ~MyMulMat();
    void init(int n, int m, int k,
              int *la, int *lb, int *lc,
              melem_t **A, melem_t **B, melem_t **C);
    uint64_t multiply();
private:
    int n, m, k;
    melem_t *A, *B, *C;
};

#if USECUDA
uint64_t cudaGemm(int, int, int, melem_t*, melem_t*, melem_t*);
#endif
