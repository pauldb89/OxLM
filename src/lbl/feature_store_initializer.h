#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/config.h"
#include "lbl/feature_matcher.h"
#include "lbl/feature_store.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FeatureStoreInitializer {
 public:
  FeatureStoreInitializer(
      const ModelData& config,
      const WordToClassIndex& index,
      const FeatureMatcher& matcher);

  void initialize(
      boost::shared_ptr<FeatureStore>& U,
      vector<boost::shared_ptr<FeatureStore>>& V,
      bool random_weights = false) const;

 private:
  const ModelData& config;
  const WordToClassIndex& index;
  const FeatureMatcher& matcher;
};

} // namespace oxlm
