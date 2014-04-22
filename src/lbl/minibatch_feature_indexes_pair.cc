#include "lbl/minibatch_feature_indexes_pair.h"

#include <boost/make_shared.hpp>

namespace oxlm {

MinibatchFeatureIndexesPair::MinibatchFeatureIndexesPair(
    const boost::shared_ptr<WordToClassIndex>& index) {
  class_indexes = boost::make_shared<MinibatchFeatureIndexes>();
  for (int i = 0; i < index->getNumClasses(); ++i) {
    word_indexes.push_back(boost::make_shared<MinibatchFeatureIndexes>());
  }
}

MinibatchFeatureIndexesPtr MinibatchFeatureIndexesPair::getClassIndexes() const {
  return class_indexes;
}

MinibatchFeatureIndexesPtr MinibatchFeatureIndexesPair::getWordIndexes(int class_id) const {
  return word_indexes[class_id];
}

void MinibatchFeatureIndexesPair::setClassIndexes(
    int feature_context_id, const vector<int>& indexes) {
  (*class_indexes)[feature_context_id] = indexes;
}

void MinibatchFeatureIndexesPair::setWordIndexes(
    int class_id, int feature_context_id, const vector<int>& indexes) {
  (*word_indexes[class_id])[feature_context_id] = indexes;
}

} // namespace oxlm
