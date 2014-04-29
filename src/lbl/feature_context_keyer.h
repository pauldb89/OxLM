#pragma once

#include "lbl/feature_context_generator.h"

namespace oxlm {

class FeatureContextKeyer {
 public:
  FeatureContextKeyer();

  FeatureContextKeyer(int hash_space, int feature_context_size);

  vector<int> getKeys(const vector<int>& context) const;

  bool operator==(const FeatureContextKeyer& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & hashSpace;
    ar & generator;
  }

  int hashSpace;
  FeatureContextGenerator generator;
  hash<FeatureContext> hash_function;
};

} // namespace oxlm
