#include "lbl/feature_matcher.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureMatcher::FeatureMatcher(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextHasher>& hasher)
    : corpus(corpus), index(index), processor(processor), hasher(hasher) {
  feature_indexes = boost::make_shared<GlobalFeatureIndexesPair>(index, hasher);
  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    vector<WordId> context = processor->extract(i);

    for (int class_context_id: hasher->getClassContextIds(context)) {
      feature_indexes->addClassIndex(class_context_id, class_id);
    }

    for (int word_context_id: hasher->getWordContextIds(class_id, context)) {
      feature_indexes->addWordIndex(class_id, word_context_id, word_class_id);
    }
  }
}

GlobalFeatureIndexesPairPtr FeatureMatcher::getGlobalFeatures() const {
  return feature_indexes;
}

MinibatchFeatureIndexesPairPtr FeatureMatcher::getMinibatchFeatures(
    const vector<int>& minibatch_indexes) const {
  MinibatchFeatureIndexesPairPtr minibatch_feature_indexes =
      boost::make_shared<MinibatchFeatureIndexesPair>(index);

  for (int i: minibatch_indexes) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    vector<WordId> context = processor->extract(i);

    for (int class_context_id: hasher->getClassContextIds(context)) {
      minibatch_feature_indexes->setClassIndexes(
          class_context_id,
          feature_indexes->getClassFeatures(class_context_id));
    }

    for (int word_context_id: hasher->getWordContextIds(class_id, context)) {
      minibatch_feature_indexes->setWordIndexes(
          class_id,
          word_context_id,
          feature_indexes->getWordFeatures(class_id, word_context_id));
    }
  }

  return minibatch_feature_indexes;
}

} // namespace oxlm
