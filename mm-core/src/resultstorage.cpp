/* 
 * resultstorage.cpp
 * 
 * Author:   Makoto Shimazu <makoto.shimaz@gmail.com>
 * URL:      https://amiq11.tumblr.com
 * LICENSE:  2-clause BSD
 * Created:  2015-05-17
 *  
 */

#include "resultstorage.h"
#include <fstream>
#include <iostream>
#include <functional>

// static
bool ResultStorage::saveTo(const std::string &filepath, const Result &result) {
  std::ofstream fout;
  fout.open(filepath.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
  if (!fout)
    return false;

  int64_t hash_result = result.hash();
  fout.write(reinterpret_cast<const char *>(&result), sizeof(Result));
  fout.write(reinterpret_cast<const char *>(&hash_result), sizeof(int64_t));
  fout.close();
  return true;
}
