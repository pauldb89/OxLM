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
  FactoredMaxentMetadata();

  FactoredMaxentMetadata(
      const boost::shared_ptr<ModelData>& config, Dict& dict);

  FactoredMaxentMetadata(
      const boost::shared_ptr<ModelData>& config, Dict& dict,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextMapper>& mapper,
      const boost::shared_ptr<BloomFilterPopulator>& populator,
      const boost::shared_ptr<FeatureMatcher>& matcher);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

  boost::shared_ptr<FeatureContextMapper> getMapper() const;

  boost::shared_ptr<BloomFilterPopulator> getPopulator() const;

  boost::shared_ptr<FeatureMatcher> getMatcher() const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FactoredMetadata>(*this);

    ar & mapper;
    ar & populator;
    ar & matcher;
  }

 protected:
  boost::shared_ptr<FeatureContextMapper> mapper;
  boost::shared_ptr<BloomFilterPopulator> populator;
  boost::shared_ptr<FeatureMatcher> matcher;
};

} // namespace oxlm
