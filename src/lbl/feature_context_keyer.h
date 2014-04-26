#pragma once

#include "lbl/feature_context_generator.h"

namespace oxlm {

class FeatureContextKeyer {
 public:
  FeatureContextKeyer();

  FeatureContextKeyer(int feature_context_size);

  vector<size_t> getKeys(const vector<int>& context) const;

  bool operator==(const FeatureContextKeyer& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & generator;
  }

  hash<FeatureContext> hash_function;
  FeatureContextGenerator generator;
};

} // namespace oxlm
