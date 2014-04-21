#include "lbl/feature_matcher.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureMatcher::FeatureMatcher(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : corpus(corpus), index(index), processor(processor), extractor(extractor) {
  feature_indexes = boost::make_shared<FeatureIndexesPair>(index->getNumClasses());
  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);

    vector<WordId> context = processor->extract(i);
    pair<vector<int>, vector<int>> feature_context_ids =
        extractor->getFeatureContextIds(class_id, context);

    for (int class_context_id: feature_context_ids.first) {
      feature_indexes->addClassIndex(class_context_id, class_id);
    }

    for (int word_context_id: feature_context_ids.second) {
      feature_indexes->addWordIndex(class_id, word_context_id, word_class_id);
    }
  }
}

FeatureIndexesPairPtr FeatureMatcher::getFeatures() const {
  return feature_indexes;
}

FeatureIndexesPairPtr FeatureMatcher::getFeatures(
    const vector<int>& minibatch_indexes) const {
  FeatureIndexesPairPtr minibatch_feature_indexes =
      boost::make_shared<FeatureIndexesPair>(index->getNumClasses());
  for (int i: minibatch_indexes) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);

    vector<WordId> context = processor->extract(i);
    pair<vector<int>, vector<int>> feature_context_ids =
        extractor->getFeatureContextIds(class_id, context);

    for (int class_context_id: feature_context_ids.first) {
      minibatch_feature_indexes->setClassIndexes(
          class_context_id,
          feature_indexes->getClassFeatures(class_context_id));
    }

    for (int word_context_id: feature_context_ids.second) {
      minibatch_feature_indexes->setWordIndexes(
          class_id,
          word_context_id,
          feature_indexes->getWordFeatures(class_id, word_context_id));
    }
  }

  return minibatch_feature_indexes;
}

} // namespace oxlm
