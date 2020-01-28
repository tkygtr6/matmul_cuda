#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "cmdline.h"
#include "dataset.h"
#include "imulmat.h"
#include "resultstorage.h"
#include "tester.h"

#if USEMPI
#  include <mpi.h>
#endif

using namespace std;

#ifndef VERSION
#  define VERSION "undefined"
#endif

#define RANK0 if (myrank_ == 0)

Tester::Tester(int argc, char *argv[], IMulMat *mm) :
    myrank_(0), type_(Dataset::free), mm_(mm), initialized_(false) {
    srand(getpid()*time(NULL));

#if USEMPI
    MPI::Init();
    myrank_ = MPI::COMM_WORLD.Get_rank();
#endif

    cmdline::parser p;
    p.add<string>("type",
                  't',
                  "type of input matrix (free, square, mv, symm, trmm, hemm)",
                  false, "free");
    p.add<string>("result",
                  'r',
                  "path to the result file",
                  false, "result.dat");
    p.add("help", 'h', "print help");
    p.add("version", 'v', "print version");

    // help
    if (!p.parse(argc, argv) || p.exist("help")) {
        print(p.error_full());
        print(p.usage());
        return;
    }

    if (p.exist("version")) {
        ostringstream os;
        os << "MulMat Tester -- version: " << VERSION << endl;
        os << "by Makoto Shimazu" << endl;
        print(os.str());
        return;
    }

    if (p.exist("type")) {
        string typeName = p.get<string>("type");
        if (typeName == "free")        type_ = Dataset::free;
        else if (typeName == "square") type_ = Dataset::square;
        else if (typeName == "mv")     type_ = Dataset::mv;
        else if (typeName == "symm")   type_ = Dataset::symm;
        else if (typeName == "trmm")   type_ = Dataset::trmm;
        else if (typeName == "hemm")   type_ = Dataset::hemm;
        else {
            ostringstream os;
            os << "Unknown type!" << endl;
            os << p.usage();
            print(os.str());
            return;
        }
    }

    // if (p.exist("result"))
    resultPath_ = p.get<string>("result");

    initialized_ = true;
}

Tester::~Tester() {
#if USEMPI
    MPI::Finalize();
#endif
}

void Tester::print(string str) {
    if (myrank_ == 0) {
        cout << str << endl;
    }
}

int Tester::_run(Dataset::DataType type) {
    print("# Run");

    uint32_t n = 0, m = 0, k = 0;
    int la, lb, lc;
    melem_t *A, *B, *C;
    print("# Prepare ");

    // Prepare dataset
    Dataset dataset;
    RANK0 {
        dataset.prepare(type, &n, &m, &k);
    }
#if USEMPI
    MPI::COMM_WORLD.Bcast(&n, 1, MPI::UNSIGNED, 0);
    MPI::COMM_WORLD.Bcast(&m, 1, MPI::UNSIGNED, 0);
    MPI::COMM_WORLD.Bcast(&k, 1, MPI::UNSIGNED, 0);
#endif
    // Allocate the spaces of matrix
    mm_->init(n, m, k, &la, &lb, &lc, &A, &B, &C);
    // Set A, B, C
    RANK0 {
        dataset.set(la, lb, lc, A, B, C);
    }

    // Measure
    print("# Run ");
    uint64_t elTime; // [us]
    elTime = mm_->multiply();

    // Check the answer
    print("# Check ");
    RANK0 {
        Result r = dataset.check(C);

        // Print result
        uint64_t flop = 2ULL * n * m * k;
        double gflops = static_cast<double>(flop) / (elTime)*1E-3;
        r.setGflops(gflops);
        ostringstream os;
        os << "# Elapsed:        "
           << static_cast<double>(elTime) / 1E3
           << " [ms]" << endl;
        os << "# Flops:          " << gflops << " [GFLOPS]" << endl;
        os << "# Wrong (Strict): " << r.getWCount() << " / " << n*m << endl;
        os << "# Wrong (Loose):  " << r.getEWCount() << " / " << n*m << endl;
        os << "# Max Wrongness:  " << r.getMaxWrong() << endl;
        print(os.str());

        if (r.getEWCount() == 0 && !ResultStorage::saveTo(resultPath_, r))
            std::cerr << "Result file cannot be generated: "
                      << "path=" << resultPath_
                      << std::endl;
    }

    return 0;
}

int Tester::run() {
    if (!initialized_)
        return 1;

    try {
        return _run(type_);
    } catch (const char *ex) {
        std::cerr << "# ERROR! :: " << ex << std::endl;
#if USEMPI
        MPI::COMM_WORLD.Abort(1);
        return 1;
#endif
    } catch (std::string ex) {
        std::cerr << "# ERROR! :: " << ex << std::endl;
#if USEMPI
        MPI::COMM_WORLD.Abort(1);
#endif
        return 1;
    }
#if USEMPI
    catch (MPI::Exception ex) {
        std::cout << "MPI::Exception" << std::endl;
        std::cout << "# ERROR! :: "
                  << ex.Get_error_string()
                  << ex.Get_error_code();
        MPI::COMM_WORLD.Abort(1);
        return 2;
    }
    catch (...) {
        std::cout << "# ERROR! :: some error has been occured..." << std::endl;
        MPI::COMM_WORLD.Abort(1);
        return 3;
    }
#endif
    return 0;
}

