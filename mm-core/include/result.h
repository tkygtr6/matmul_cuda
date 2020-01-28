/* 
 * result.h
 * 
 * Author:   Makoto Shimazu <makoto.shimaz@gmail.com>
 * URL:      https://amiq11.tumblr.com               
 * Created:  2015-05-17                              
 *  
 */

#pragma once


#include <stdint.h>

class Result {
private:
  int wcount_, ewcount_;
  double maxwrong_;
  double gflops_;
public:
  Result(int w, int ew, float mw) {
    wcount_ = w; ewcount_ = ew;
    maxwrong_ = mw;
  }
  void setGflops(double gflops) {
    gflops_ = gflops;
  }
  int getWCount() const { return wcount_; }
  int getEWCount() const { return ewcount_; }
  double getMaxWrong() const { return maxwrong_; }
  double getGflops() const { return gflops_; }
  int64_t hash() const;
};
