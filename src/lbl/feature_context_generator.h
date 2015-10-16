#pragma once

#include "lbl/feature_context.h"
#include "lbl/hash_builder.h"

namespace oxlm {

class FeatureContextGenerator {
 public:
  FeatureContextGenerator();

  FeatureContextGenerator(size_t feature_context_size);

  vector<FeatureContext> getFeatureContexts(const vector<int>& context) const;

  vector<Hash> getFeatureContextHashes(const vector<int>& context) const;

  bool operator==(const FeatureContextGenerator& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & featureContextSize;
    //TODO(pauldb): Serialize hashBuidler.
  }

  size_t featureContextSize;
  HashBuilder hashBuilder;
};

} // namespace oxlm
