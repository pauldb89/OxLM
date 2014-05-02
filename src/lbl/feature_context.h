#pragma once

#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class FeatureContext {
 public:
  FeatureContext();

  FeatureContext(const vector<int>& data);

  bool operator==(const FeatureContext& feature_context) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & data;
  }

 public:
  vector<int> data;
};

} // namespace oxlm

namespace std {

template<> struct hash<oxlm::FeatureContext> {
  inline size_t operator()(const oxlm::FeatureContext& feature_context) const {
    return oxlm::MurmurHash(feature_context.data);
  }
};

} // namespace std
