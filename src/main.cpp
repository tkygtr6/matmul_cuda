#include <memory>
#include <tester.h>

#include "mymulmat.h"

int main(int argc, char *argv[]) {
    MyMulMat mm;
    Tester test(argc, argv, &mm);
    return test.run();
}
