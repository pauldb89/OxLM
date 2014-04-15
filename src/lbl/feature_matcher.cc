#include "lbl/feature_matcher.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureMatcher::FeatureMatcher(
    const Corpus& corpus, const WordToClassIndex& index,
    const ContextProcessor& processor,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : corpus(corpus), index(index), processor(processor), extractor(extractor) {
  feature_indexes = boost::make_shared<FeatureIndexesPair>(index.getNumClasses());
  for (size_t i = 0; i < corpus.size(); ++i) {
    int word_id = corpus[i];
    int class_id = index.getClass(word_id);
    int word_class_id = index.getWordIndexInClass(word_id);

    vector<WordId> context = processor.extract(i);
    vector<FeatureContextId> feature_context_ids =
        extractor->getFeatureContextIds(context);
    for (const FeatureContextId& feature_context_id: feature_context_ids) {
      feature_indexes->addClassIndex(feature_context_id, class_id);
      feature_indexes->addWordIndex(class_id, feature_context_id, word_class_id);
    }
  }
}

FeatureIndexesPairPtr FeatureMatcher::getFeatures() const {
  return feature_indexes;
}

FeatureIndexesPairPtr FeatureMatcher::getFeatures(
    const vector<int>& minibatch_indexes) const {
  FeatureIndexesPairPtr minibatch_feature_indexes =
      boost::make_shared<FeatureIndexesPair>(index.getNumClasses());
  for (int i: minibatch_indexes) {
    int word_id = corpus[i];
    int class_id = index.getClass(word_id);
    int word_class_id = index.getWordIndexInClass(word_id);

    vector<WordId> context = processor.extract(i);
    vector<FeatureContextId> feature_context_ids =
        extractor->getFeatureContextIds(context);
    for (const FeatureContextId& feature_context_id: feature_context_ids) {
      minibatch_feature_indexes->setClassIndexes(
          feature_context_id,
          feature_indexes->getClassFeatures(feature_context_id));
      minibatch_feature_indexes->setWordIndexes(
          class_id,
          feature_context_id,
          feature_indexes->getWordFeatures(class_id, feature_context_id));
    }
  }
  return minibatch_feature_indexes;
}

} // namespace oxlm
