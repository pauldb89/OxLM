#pragma once

#include "lbl/feature_context.h"

namespace oxlm {

class FeatureContextGenerator {
 public:
  FeatureContextGenerator();

  FeatureContextGenerator(size_t feature_context_size);

  vector<FeatureContext> getFeatureContexts(const vector<int>& context) const;

  bool operator==(const FeatureContextGenerator& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & featureContextSize;
  }

  size_t featureContextSize;
};

} // namespace oxlm
