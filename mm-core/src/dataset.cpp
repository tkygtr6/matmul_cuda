//! g++ -std=c++11 dataset.cpp -fsyntax-only
#include "dataset.h"
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <cstring>

using namespace std;

const string Dataset::dataDir      = "./data/";
const string Dataset::fileListPath = "./data/list.txt";
typedef map< Dataset::DataType, vector< string > > fileListMap_t;
fileListMap_t Dataset::fileListMap;
typedef map< string, Dataset::DataType > typeMap_t;
typeMap_t Dataset::typeMap;

Dataset::Dataset()
{
    if (typeMap.empty()) createTypeMap();
    if (fileListMap.empty()) createFileListMap();
}

Dataset::~Dataset()
{
}

void Dataset::createFileListMap()
{
    ifstream listfs(fileListPath.c_str());
    if (listfs.fail()) {
        throw string("\"") + fileListPath + "\" is not found!";
    }

    string type_str;
    // Decode first line
    listfs >> type_str;
    if (typeMap.count(type_str) == 0) {
        throw string("Unknown DataType ") + type_str;
    }
    DataType type = typeMap[type_str];
    while (!listfs.eof()) {
        string line;
        listfs >> line;
        if ( line == "" ) break;
        // Check whether the line matches the header lines
        if (typeMap.count(line) != 0) {
            // Set new type
            type_str = line;
            type = typeMap[type_str];
        } else {
            fileListMap[type].push_back(line);
        }
    }
}

void Dataset::createTypeMap()
{
    typeMap["<free>"]   = free;
    typeMap["<square>"] = square;
    typeMap["<mv>"]     = mv;
    typeMap["<symm>"]   = symm;
    typeMap["<trmm>"]   = trmm;
    typeMap["<hemm>"]   = hemm;
}

void Dataset::prepare(DataType type,
                      uint32_t *n, uint32_t *m, uint32_t *k)
{
    // prepare
    // random_device seed_gen;
    // mt19937 random(seed_gen());
    vector<string> files = fileListMap[type];
    if (files.size() == 0) {
        throw "There is no file in this type";
    }
    string file = files[rand()%files.size()];
    ifs.open((dataDir + file).c_str());
    cout << "# " << file << " is selected!" << endl;
    if (ifs.fail()) {
        throw "FAILED TO OPEN!";
    }

    // read n, m, k
    ifs.read((char*)n, sizeof(uint32_t));
    ifs.read((char*)m, sizeof(uint32_t));
    ifs.read((char*)k, sizeof(uint32_t));
    cout << "# n=" << *n << " m=" << *m << " k=" << *k << endl;
    this->n = *n; this->m = *m; this->k = *k;
}


static bool isLarge(uint32_t n, uint32_t m, uint32_t k) {
    typedef unsigned long long ull;
    return (ull)n*(ull)k + (ull)k*(ull)m + (ull)n*(ull)m >= 3*(1UL<<22);
}

static melem_t *gen_nullmatrix(uint32_t n, uint32_t m)
{
    melem_t * ret = (melem_t*)malloc(sizeof(melem_t) * n * m );
    if(!ret){
        perror("malloc failed.\n");
        exit(EXIT_FAILURE);
    }
    return ret;
}  



void Dataset::set(int la, int lb, int lc,
                  melem_t *A, melem_t *B, melem_t *C)
{
    if (isLarge(n, m, k)) {
        uint32_t pars[7];
        ifs.read((char*)pars, sizeof(uint32_t)*7);
        this->tn = pars[0], this->tm = pars[1], this->tk = pars[2];
        this->x  = pars[4], this->y  = pars[5], this->z  = pars[6];
        melem_t *   a = gen_nullmatrix(tn, tk);
        melem_t *  ba = gen_nullmatrix( x, tk);
        melem_t *  ra = gen_nullmatrix(tn,  y);
        melem_t * rba = gen_nullmatrix( x,  y);

        melem_t *   b = gen_nullmatrix(tk, tm);
        melem_t *  bb = gen_nullmatrix( y, tm);
        melem_t *  rb = gen_nullmatrix(tk,  z);
        melem_t * rbb = gen_nullmatrix( y,  z);

        ifs.read((char*)  a, sizeof(melem_t) * tn * tk);
        ifs.read((char*) ba, sizeof(melem_t) *  x * tk);
        ifs.read((char*) ra, sizeof(melem_t) * tn *  y);
        ifs.read((char*)rba, sizeof(melem_t) *  x *  y);

        ifs.read((char*)  b, sizeof(melem_t) * tk * tm);
        ifs.read((char*) bb, sizeof(melem_t) *  y * tm);
        ifs.read((char*) rb, sizeof(melem_t) * tk *  z);
        ifs.read((char*)rbb, sizeof(melem_t) *  y *  z);

        uint32_t i, ii, jj;
        for ( ii = 0; ii+tn < n; ii+=tn ) {
            for ( jj = 0; jj+tk < k; jj+=tk ) 
                for ( i = 0; i < tn; i++ )
                    memcpy(&A[(i+ii)*la + jj], &a[i*tk], sizeof(melem_t)*tk);
            for ( i = 0; i < tn; i++ ) 
                memcpy(&A[(i+ii)*la + jj], &ra[i*y], sizeof(melem_t)*y);
        }
        for ( jj = 0; jj+tk < k; jj+=tk ) 
            for ( i = 0; i < x; i++ ) 
                memcpy(&A[(i+ii)*la + jj], &ba[i*tk], sizeof(melem_t)*tk);
        for ( i = 0; i < x; i++ ) 
            memcpy(&A[(i+ii)*la + jj], &rba[i*y], sizeof(melem_t)*y);

        for ( ii = 0; ii+tk < k; ii+=tk ) {
            for ( jj = 0; jj+tm < m; jj+=tm ) 
                for ( i = 0; i < tk; i++ )
                    memcpy(&B[(i+ii)*lb + jj], &b[i*tm], sizeof(melem_t)*tm);
            for ( i = 0; i < tk; i++ )
                memcpy(&B[(i+ii)*lb + jj], &rb[i*z], sizeof(melem_t)*z);
        }
        for ( jj = 0; jj+tm < m; jj+=tm ) 
            for ( i = 0; i < y; i++ )
                memcpy(&B[(i+ii)*lb + jj], &bb[i*tm], sizeof(melem_t)*tm);
        for ( i = 0; i < y; i++ ) 
            memcpy(&B[(i+ii)*lb + jj], &rbb[i*z], sizeof(melem_t)*z);

        bzero(C, sizeof(melem_t)*n*m);

        delete a; delete ba; delete ra; delete rba;
        delete b; delete bb; delete rb; delete rbb;

    } else {
        // Fill in A, B, C
        for ( uint32_t i = 0; i < n; i++ ) {
            ifs.read((char*)(A+i*la), sizeof(melem_t)*k);
        }
        for ( uint32_t i = 0; i < k; i++ ) {
            ifs.read((char*)(B+i*lb), sizeof(melem_t)*m);
        }
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < m; j++) {
                C[i*m+j] = 0;
            }
        }
    }
    this->la = la; this->lb = lb; this->lc = lc;
}

namespace {
  static inline void check_one(int *wcount, int *ewcount, double *maxwrong, melem_t C, melem_t ans, int i, int j, double delta, double epsilon) {
    double diff = fabs(C - ans);
    if ( !isfinite(diff) || diff > delta ) {
#if PRINTWRONG
      cerr << "### WRONG: "
           << "C(" << i << ", " << j << ") != "
           << "Ans(" << i << ", " << j << ") :: "
           << "C = " << C << ", "
           << "Ans = " << ans << endl;
#endif
      (*wcount)++;
    }
    if ( !isfinite(diff) || diff > epsilon ) {
      (*ewcount)++;
    }
    if ( diff > *maxwrong ) {
      *maxwrong = diff;
    }
  }
}  // namespace

Result Dataset::check(melem_t *C)
{
    int wcount = 0;             // # of wrong answer
    int ewcount = 0;            // # of truly wrong answer
    double maxwrong = 0;
    double delta   = pow(2,(log2f(k)+7*2)-23); // MAX*2^(-2) (float pricision is 2^23, max of each value is +/-2^7)
    double epsilon = pow(2,(log2f(k)+7*2)-20); // MAX*2^(1)
    cout << "# (Delta(Strict) = " << delta   << ")" << endl;
    cout << "# (Delta(Loose)  = " << epsilon << ")" << endl;
    if(isLarge(n, m, k)){
        melem_t *   c = gen_nullmatrix(tn, tm);
        melem_t *  bc = gen_nullmatrix( x, tm);
        melem_t *  rc = gen_nullmatrix(tn,  z);
        melem_t * rbc = gen_nullmatrix( x,  z);
        ifs.read((char*)  c, sizeof(melem_t) * tn * tm);
        ifs.read((char*) bc, sizeof(melem_t) *  x * tm);
        ifs.read((char*) rc, sizeof(melem_t) * tn *  z);
        ifs.read((char*)rbc, sizeof(melem_t) *  x *  z);

        uint32_t i, j, ii, jj;
        for ( ii = 0; ii+tn < n; ii+=tn ) {
            for ( jj = 0; jj+tm < m; jj+=tm )
                for ( i = 0; i < tn; i++ )
                    for ( j = 0; j < tm; j++ ) {
                      check_one(&wcount, &ewcount, &maxwrong, C[(ii+i)*lc + (jj+j)], c[i*tm+j], ii+i, jj+j, delta, epsilon);
                    }
            for ( i = 0; i < tn; i++ )
                for ( j = 0; j < z; j++ ) {
                      check_one(&wcount, &ewcount, &maxwrong, C[(ii+i)*lc + (jj+j)], rc[i*z+j], ii+i, jj+j, delta, epsilon);
                }
        }
        for ( jj = 0; jj+tm < m; jj+=tm )
            for ( i = 0; i < x; i++ )
                for ( j = 0; j < tm; j++ ) {
                      check_one(&wcount, &ewcount, &maxwrong, C[(ii+i)*lc + (jj+j)], bc[i*tm+j], ii+i, jj+j, delta, epsilon);
                }
        for ( i = 0; i < x; i++ )
            for ( j = 0; j < z; j++ ) {
              check_one(&wcount, &ewcount, &maxwrong, C[(ii+i)*lc + (jj+j)], rbc[i*z+j], ii+i, jj+j, delta, epsilon);
            }
        delete c; delete bc; delete rc; delete rbc;
    }else{
        // Check Answer
        melem_t *ans = new melem_t[n*m]();
        ifs.read((char*)ans, sizeof(melem_t)*n*m);
        for ( uint32_t i = 0; i < n; i++ ) {
            for ( uint32_t j = 0; j < m; j++ ) {
              check_one(&wcount, &ewcount, &maxwrong, C[i*lc + j], ans[i*m+j], i, j, delta, epsilon);
            }
        }
    }
    return Result(wcount, ewcount, maxwrong);
}
