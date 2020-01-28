#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "dataset.h"
#include "imulmat.h"

class Tester
{
public:
    Tester(int argc, char *argv[], IMulMat *mm);
    ~Tester();
    int run();
private:
    int _run(Dataset::DataType);
    void print(std::string str);

    int myrank_;
    Dataset::DataType type_;
    IMulMat *mm_;
    bool initialized_;
    std::string resultPath_;
};
