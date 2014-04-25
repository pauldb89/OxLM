#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/config.h"
#include "lbl/feature_context_hasher.h"
#include "lbl/feature_matcher.h"
#include "lbl/global_feature_store.h"
#include "lbl/minibatch_feature_store.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FeatureStoreInitializer {
 public:
  FeatureStoreInitializer(
      const ModelData& config,
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index);

  void initialize(
      boost::shared_ptr<GlobalFeatureStore>& U,
      vector<boost::shared_ptr<GlobalFeatureStore>>& V) const;

  void initialize(
      boost::shared_ptr<MinibatchFeatureStore>& U,
      vector<boost::shared_ptr<MinibatchFeatureStore>>& V,
      const vector<int>& minibatch_indices) const;

 private:
  ModelData config;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FeatureContextHasher> hasher;
  boost::shared_ptr<FeatureMatcher> matcher;
};

} // namespace oxlm
