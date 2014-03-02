#pragma once

#include <vector>

#include <boost/functional/hash.hpp>

using namespace std;

namespace oxlm {

struct Feature {
  Feature(char feature_type, const vector<int>& data);

  bool operator==(const Feature& feature) const;

  char feature_type;
  vector<int> data;
};

} // namespace oxlm

namespace std {

template<> struct hash<oxlm::Feature> {
  inline size_t operator()(const oxlm::Feature& feature) const {
    size_t result = 0;
    boost::hash_combine(result, feature.feature_type);
    boost::hash_combine(result, feature.data);
    return result;
  }
};

} // namespace std
