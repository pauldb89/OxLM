#pragma once

#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

namespace oxlm {

class FeatureContext {
 public:
  FeatureContext();

  FeatureContext(char feature_type, const vector<int>& data);

  bool operator==(const FeatureContext& feature_context) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & feature_type;
    ar & data;
  }

 public:
  char feature_type;
  vector<int> data;
};

typedef vector<pair<vector<FeatureContext>, int>> MatchingContexts;

} // namespace oxlm

namespace std {

template<> struct hash<oxlm::FeatureContext> {
  inline size_t operator()(const oxlm::FeatureContext& feature_context) const {
    size_t result = 0;
    boost::hash_combine(result, feature_context.feature_type);
    boost::hash_combine(result, feature_context.data);
    return result;
  }
};

} // namespace std
