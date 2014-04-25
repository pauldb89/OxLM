#include "lbl/global_feature_indexes_pair.h"

#include <boost/make_shared.hpp>

namespace oxlm {

GlobalFeatureIndexesPair::GlobalFeatureIndexesPair(
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextHasher>& hasher) {
  class_indexes = boost::make_shared<GlobalFeatureIndexes>(
      hasher->getNumClassContexts());
  for (int i = 0; i < index->getNumClasses(); ++i) {
    word_indexes.push_back(boost::make_shared<GlobalFeatureIndexes>(
        hasher->getNumWordContexts(i)));
  }
}

GlobalFeatureIndexesPtr GlobalFeatureIndexesPair::getClassIndexes() const {
  return class_indexes;
}

GlobalFeatureIndexesPtr GlobalFeatureIndexesPair::getWordIndexes(int class_id) const {
  return word_indexes[class_id];
}

vector<int> GlobalFeatureIndexesPair::getClassFeatures(
    int feature_context_id) const {
  return class_indexes->at(feature_context_id);
}

vector<int> GlobalFeatureIndexesPair::getWordFeatures(
    int class_id, int feature_context_id) const {
  return word_indexes[class_id]->at(feature_context_id);
}

void GlobalFeatureIndexesPair::addClassIndex(
    int feature_context_id, int class_id) {
  vector<int>& indexes = class_indexes->at(feature_context_id);
  if (find(indexes.begin(), indexes.end(), class_id) == indexes.end()) {
    indexes.reserve(indexes.size() + 1);
    indexes.push_back(class_id);
  }
}

void GlobalFeatureIndexesPair::addWordIndex(
    int class_id, int feature_context_id, int word_class_id) {
  vector<int>& indexes = word_indexes[class_id]->at(feature_context_id);
  if (find(indexes.begin(), indexes.end(), word_class_id) == indexes.end()) {
    indexes.reserve(indexes.size() + 1);
    indexes.push_back(word_class_id);
  }
}

} // namespace oxlm
