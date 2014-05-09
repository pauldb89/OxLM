#pragma once

#include "lbl/feature_context.h"

namespace oxlm {

class FeatureContextKeyer {
 public:
  FeatureContextKeyer();

  FeatureContextKeyer(int hash_space);

  int getKey(const FeatureContext& feature_context) const;

  bool operator==(const FeatureContextKeyer& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & hashSpace;
  }

  int hashSpace;
  hash<FeatureContext> hash_function;
};

} // namespace oxlm
