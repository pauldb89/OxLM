#include "lbl/factored_maxent_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/collision_counter.h"

namespace oxlm {

FactoredMaxentMetadata::FactoredMaxentMetadata() {}

FactoredMaxentMetadata::FactoredMaxentMetadata(
    const boost::shared_ptr<ModelData>& config, Dict& dict)
    : FactoredMetadata(config, dict) {}

FactoredMaxentMetadata::FactoredMaxentMetadata(
    const boost::shared_ptr<ModelData>& config, Dict& dict,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextMapper>& mapper,
    const boost::shared_ptr<BloomFilterPopulator>& populator,
    const boost::shared_ptr<FeatureMatcher>& matcher)
    : FactoredMetadata(config, dict, index),
      mapper(mapper), populator(populator), matcher(matcher) {}

void FactoredMaxentMetadata::initialize(
    const boost::shared_ptr<Corpus>& corpus) {
  FactoredMetadata::initialize(corpus);

  if (!config->hash_space || config->filter_contexts) {
    int context_width = config->ngram_order - 1;
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, context_width);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(
            config->feature_context_size);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(
            corpus, index, processor, generator, config->max_ngrams,
            config->min_ngram_freq);
    cout << "Done creating the n-gram filter..." << endl;

    mapper = boost::make_shared<FeatureContextMapper>(
        corpus, index, processor, generator, filter);
    cout << "Done creating the feature context mapper..." << endl;
    if (config->filter_contexts && config->filter_error_rate) {
      populator = boost::make_shared<BloomFilterPopulator>(
          corpus, index, mapper, config);
      cout << "Done creating the Bloom filter populator..." << endl;
    } else {
      matcher = boost::make_shared<FeatureMatcher>(
          corpus, index, processor, generator, filter, mapper);
      cout << "Done creating the feature matcher..." << endl;
    }
  }

  if (config->hash_space > 0 && config->count_collisions) {
      CollisionCounter counter(
          corpus, index, mapper, matcher, populator, config);
      counter.count();
  }
}

boost::shared_ptr<FeatureContextMapper> FactoredMaxentMetadata::getMapper() const {
  return mapper;
}

boost::shared_ptr<BloomFilterPopulator> FactoredMaxentMetadata::getPopulator() const {
  return populator;
}

boost::shared_ptr<FeatureMatcher> FactoredMaxentMetadata::getMatcher() const {
  return matcher;
}

} // namespace oxlm
