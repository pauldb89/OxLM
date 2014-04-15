#include "lbl/feature_indexes_pair.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureIndexesPair::FeatureIndexesPair(int num_classes) {
  class_indexes = boost::make_shared<FeatureIndexes>();
  for (int i = 0; i < num_classes; ++i) {
    word_indexes.push_back(boost::make_shared<FeatureIndexes>());
  }
}

FeatureIndexesPtr FeatureIndexesPair::getClassIndexes() const {
  return class_indexes;
}

FeatureIndexesPtr FeatureIndexesPair::getWordIndexes(int class_id) const {
  return word_indexes[class_id];
}

unordered_set<int> FeatureIndexesPair::getClassFeatures(
    int feature_context_id) const {
  return class_indexes->at(feature_context_id);
}

unordered_set<int> FeatureIndexesPair::getWordFeatures(
    int class_id, int feature_context_id) const {
  return word_indexes[class_id]->at(feature_context_id);
}

void FeatureIndexesPair::addClassIndex(
    FeatureContextId feature_context_id, int class_id) {
  (*class_indexes)[feature_context_id].insert(class_id);
}

void FeatureIndexesPair::addWordIndex(
    int class_id, FeatureContextId feature_context_id, int word_class_id) {
  (*word_indexes[class_id])[feature_context_id].insert(word_class_id);
}

void FeatureIndexesPair::setClassIndexes(
    FeatureContextId feature_context_id,
    const unordered_set<int>& indexes) {
  (*class_indexes)[feature_context_id] = indexes;
}

void FeatureIndexesPair::setWordIndexes(
    int class_id,
    FeatureContextId feature_context_id,
    const unordered_set<int>& indexes) {
  (*word_indexes[class_id])[feature_context_id] = indexes;
}

} // namespace oxlm
