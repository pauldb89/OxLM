#pragma once

#include <unordered_map>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/thread/tss.hpp>

#include "lbl/utils.h"

using namespace std;

namespace oxlm {

typedef unordered_map<vector<int>, Real, boost::hash<vector<int>>> ContextMap;

/**
 * Thread safe cache mapping contexts to normalizers.
 *
 * Thread safety is required by moses.
 */
class ContextCache {
 public:
  pair<Real, bool> get(const vector<int>& context);

  void set(const vector<int>& context, Real value);

  void clear();

 private:
  boost::thread_specific_ptr<ContextMap> cache;
};

} // namespace oxlm
