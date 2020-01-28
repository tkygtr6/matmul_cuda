/* 
 * resultstorage.h
 * 
 * Author:   Makoto Shimazu <makoto.shimaz@gmail.com>
 * URL:      https://amiq11.tumblr.com               
 * Created:  2015-05-17                              
 *  
 */

#pragma once

#include <string>
#include "dataset.h"
#include "result.h"

class ResultStorage {
 public:
  static bool saveTo(const std::string &filepath, const Result &result);
};
