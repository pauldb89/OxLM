#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/config.h"
#include "lbl/feature_indexes_pair.h"
#include "lbl/feature_matcher.h"
#include "lbl/feature_store.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FeatureStoreInitializer {
 public:
  FeatureStoreInitializer(
      const ModelData& config,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureMatcher>& matcher);

  void initialize(
      boost::shared_ptr<FeatureStore>& U,
      vector<boost::shared_ptr<FeatureStore>>& V,
      bool random_weights = false) const;

  void initialize(
      boost::shared_ptr<FeatureStore>& U,
      vector<boost::shared_ptr<FeatureStore>>& V,
      const vector<int>& minibatch_indices, bool random_weights = false) const;

 private:
  void initializeUnconstrainedStores(
      boost::shared_ptr<FeatureStore>& U,
      vector<boost::shared_ptr<FeatureStore>>& V) const;

  void initializeSparseStores(
      boost::shared_ptr<FeatureStore>& U,
      vector<boost::shared_ptr<FeatureStore>>& V,
      FeatureIndexesPairPtr feature_indexes_pair,
      bool random_weights) const;

  ModelData config;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FeatureMatcher> matcher;
};

} // namespace oxlm
