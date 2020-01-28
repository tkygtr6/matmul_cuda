#include "mymulmat.h"

#include <iostream>

#include <sys/time.h>
uint64_t getus() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1E6 + tv.tv_usec;
}

MyMulMat::MyMulMat() {
    std::cout << "mymul constructed" << std::endl;
}

MyMulMat::~MyMulMat() {
    std::cout << "mymul destructed" << std::endl;
    delete[] A;
    delete[] B;
    delete[] C;
}
void MyMulMat::init(int n, int m, int k,
          int *la, int *lb, int *lc,
          melem_t **A, melem_t **B, melem_t **C) {
    std::cout << "mymul init" << std::endl;
    *la = k; *lb = m; *lc = m;
    *A = new melem_t[size_t(n)*k]();
    *B = new melem_t[size_t(k)*m]();
    *C = new melem_t[size_t(n)*m]();
    this->n = n; this->m = m; this->k = k;
    this->A = *A; this->B = *B; this->C = *C;
    return;
}

uint64_t MyMulMat::multiply() {
    uint64_t elTime;
    std::cout << "mymul multiply" << std::endl;
#if USECUDA
    elTime = cudaGemm(n, m, k, A, B, C);
#else
    uint64_t before, after;
    before = getus();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < k; l++ ) {
                C[i*m+j] += A[i*k+l] * B[l*m+j];
            }
        }
    }
    after = getus();
    elTime = after - before;
#endif
    return elTime;
}
