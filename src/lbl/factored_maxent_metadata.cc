#include "lbl/factored_maxent_metadata.h"

#include <boost/make_shared.hpp>

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
    const boost::shared_ptr<FeatureMatcher>& matcher)
    : FactoredMetadata(config, vocab, index), matcher(matcher) {}

void FactoredMaxentMetadata::initialize(
  const boost::shared_ptr<Corpus>& corpus) {
FactoredMetadata::initialize(corpus);

  int context_width = config->ngram_order - 1;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, context_width);
  boost::shared_ptr<FeatureContextGenerator> generator =
      boost::make_shared<FeatureContextGenerator>(
          config->feature_context_size);

  auto start_time = GetTime();
  boost::shared_ptr<NGramFilter> filter;
  if (config->ngram_file.size()) {
    filter = boost::make_shared<NGramFilter>(config->ngram_file);
  } else {
    filter = boost::make_shared<NGramFilter>();
  }
  cout << "Done creating the n-gram filter..." << endl;
  cout << "Creating the n-gram filter took " << GetDuration(start_time, GetTime())
       << " seconds..." << endl;

  start_time = GetTime();
  matcher = boost::make_shared<FeatureMatcher>(
      corpus, index, processor, generator, filter);
  cout << "Done creating the feature matcher..." << endl;
  cout << "Creating the feature matcher took " << GetDuration(start_time, GetTime())
       << " seconds..." << endl;
}

boost::shared_ptr<FeatureMatcher> FactoredMaxentMetadata::getMatcher() const {
  return matcher;
}

} // namespace oxlm
