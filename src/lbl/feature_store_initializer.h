#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/bloom_filter_populator.h"
#include "lbl/config.h"
#include "lbl/feature_context_mapper.h"
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
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextMapper>& mapper,
      const boost::shared_ptr<FeatureMatcher>& matcher,
      const boost::shared_ptr<BloomFilterPopulator>& popualator);

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
  boost::shared_ptr<FeatureContextMapper> mapper;
  boost::shared_ptr<FeatureMatcher> matcher;
  boost::shared_ptr<BloomFilterPopulator> populator;
};

} // namespace oxlm
