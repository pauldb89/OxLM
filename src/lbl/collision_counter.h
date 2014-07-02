#pragma once

#include <unordered_set>

#include "lbl/bloom_filter_populator.h"
#include "lbl/config.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_matcher.h"
#include "lbl/ngram.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class CollisionCounter {
 public:
  CollisionCounter(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextMapper>& mapper,
      const boost::shared_ptr<FeatureMatcher>& matcher,
      const boost::shared_ptr<BloomFilterPopulator>& populator,
      const boost::shared_ptr<ModelData>& config);

  int count() const;

 private:
  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FeatureContextMapper> mapper;
  boost::shared_ptr<FeatureMatcher> matcher;
  boost::shared_ptr<BloomFilterPopulator> populator;
  unordered_set<NGram> observedClassQueries;
  vector<unordered_set<NGram>> observedWordQueries;
  unordered_set<int> observedKeys;
};

} // namespace oxlm
