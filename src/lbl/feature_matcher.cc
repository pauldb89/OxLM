#include "lbl/feature_matcher.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureMatcher::FeatureMatcher() {}

FeatureMatcher::FeatureMatcher(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextGenerator>& generator,
    const boost::shared_ptr<NGramFilter>& filter,
    const boost::shared_ptr<FeatureContextMapper>& mapper)
    : corpus(corpus), index(index), generator(generator), mapper(mapper) {
  featureIndexes = boost::make_shared<GlobalFeatureIndexesPair>(index, mapper);
  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    vector<WordId> context = processor->extract(i);

    vector<FeatureContext> feature_contexts =
        generator->getFeatureContexts(context);
    // Add feature indexes only for the topmost max_ngrams ngrams.
    feature_contexts = filter->filter(word_id, class_id, feature_contexts);

    for (int context_id: mapper->getClassContextIds(feature_contexts)) {
      featureIndexes->addClassIndex(context_id, class_id);
    }

    for (int context_id: mapper->getWordContextIds(class_id, feature_contexts)) {
      featureIndexes->addWordIndex(class_id, context_id, word_class_id);
    }
  }
}

GlobalFeatureIndexesPairPtr FeatureMatcher::getGlobalFeatures() const {
  return featureIndexes;
}

} // namespace oxlm
