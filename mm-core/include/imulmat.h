#pragma once

#include "dataset.h"
#include <cstdint>

class IMulMat {
public:
    virtual void init(int n, int m, int k,
                      int *la, int *lb, int *lc,
                      melem_t **A, melem_t **B, melem_t **C) = 0;
    virtual uint64_t multiply() = 0;
};
