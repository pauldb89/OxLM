#include "lbl/feature_matcher.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureMatcher::FeatureMatcher() {}

FeatureMatcher::FeatureMatcher(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextGenerator>& generator,
    const boost::shared_ptr<NGramFilter>& filter) {
  featureIndexes = boost::make_shared<FeatureIndexesPair>(index);

  size_t num_processed = 0;
  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    vector<WordId> context = processor->extract(i);

    vector<Hash> context_hashes = generator->getFeatureContextHashes(context);
    // Add feature indexes only for the topmost max_ngrams ngrams.
    context_hashes = filter->filter(word_id, class_id, context_hashes);

    for (Hash context_hash: context_hashes) {
      featureIndexes->addClassIndex(context_hash, class_id);
    }
    for (Hash context_hash: context_hashes) {
      featureIndexes->addWordIndex(class_id, context_hash, word_class_id);
    }

    ++num_processed;
    if (num_processed % 1000000 == 0) {
      cout << ".";
      if (num_processed % 100000000 == 0) {
        cout << " [" << num_processed << "]" << endl;
      }
    }
  }
  cout << endl;
}

FeatureIndexesPairPtr FeatureMatcher::getFeatureIndexes() const {
  return featureIndexes;
}

} // namespace oxlm
