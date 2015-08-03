#include "lbl/factored_maxent_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/collision_counter.h"

namespace oxlm {

FactoredMaxentMetadata::FactoredMaxentMetadata() {}

FactoredMaxentMetadata::FactoredMaxentMetadata(
    const boost::shared_ptr<ModelData>& config,
    boost::shared_ptr<Vocabulary>& vocab)
    : FactoredMetadata(config, vocab) {}

FactoredMaxentMetadata::FactoredMaxentMetadata(
    const boost::shared_ptr<ModelData>& config,
    boost::shared_ptr<Vocabulary>& vocab,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextMapper>& mapper,
    const boost::shared_ptr<FeatureMatcher>& matcher)
    : FactoredMetadata(config, vocab, index),
      mapper(mapper), matcher(matcher) {}

void FactoredMaxentMetadata::initialize(
  const boost::shared_ptr<Corpus>& corpus) {
FactoredMetadata::initialize(corpus);

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
  matcher = boost::make_shared<FeatureMatcher>(
      corpus, index, processor, generator, filter, mapper);
  cout << "Done creating the feature matcher..." << endl;

  if (config->hash_space > 0 && config->count_collisions) {
      CollisionCounter counter(
          corpus, index, mapper, matcher, config);
      counter.count();
  }
}

boost::shared_ptr<FeatureContextMapper> FactoredMaxentMetadata::getMapper() const {
  return mapper;
}

boost::shared_ptr<FeatureMatcher> FactoredMaxentMetadata::getMatcher() const {
  return matcher;
}

} // namespace oxlm
