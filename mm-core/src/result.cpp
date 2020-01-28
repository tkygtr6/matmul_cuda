/* 
 * result.cpp
 * 
 * Author:   Makoto Shimazu <makoto.shimaz@gmail.com>
 * URL:      https://amiq11.tumblr.com               
 * Created:  2015-05-17                              
 *  
 */

#include "result.h"

namespace {
template <typename T>
int64_t hash(T x) {
  int64_t value = 0;
  char *bytes = reinterpret_cast<char *>(&x);
  for (uint64_t i = 0; i < sizeof(T); i++)
    value = bytes[i] + value * 37;
  return value;
}
}  // namespace

int64_t Result::hash() const {
  int64_t values[] = {
    ::hash(wcount_),
    ::hash(ewcount_),
    ::hash(maxwrong_),
    ::hash(gflops_)
  };
  int64_t ret = 0;
  for (uint32_t i = 0; i < sizeof(values)/sizeof(int64_t); i++)
    ret = values[i] + ret * 37;
  return ret;
}
