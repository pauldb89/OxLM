#pragma once

#include "lbl/bloom_filter_populator.h"
#include "lbl/factored_metadata.h"
#include "lbl/feature_context_generator.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_matcher.h"
#include "lbl/ngram_filter.h"

namespace oxlm {

class FactoredMaxentMetadata : public FactoredMetadata {
 public:
  FactoredMaxentMetadata(const ModelData& config, Dict& dict);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

 protected:
  boost::shared_ptr<FeatureContextMapper> mapper;
  boost::shared_ptr<BloomFilterPopulator> populator;
  boost::shared_ptr<FeatureMatcher> matcher;
};

} // namespace oxlm
