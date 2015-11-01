#include "lbl/feature_indexes_pair.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureIndexesPair::FeatureIndexesPair() {}

FeatureIndexesPair::FeatureIndexesPair(const boost::shared_ptr<WordToClassIndex>& index) {
  classIndex = boost::make_shared<FeatureIndex>();
  for (int i = 0; i < index->getNumClasses(); ++i) {
    wordIndexes.push_back(boost::make_shared<FeatureIndex>());
  }
}

FeatureIndexPtr FeatureIndexesPair::getClassIndex() const {
  return classIndex;
}

FeatureIndexPtr FeatureIndexesPair::getWordIndexes(int class_id) const {
  return wordIndexes[class_id];
}

vector<int> FeatureIndexesPair::getClassFeatures(Hash h) const {
  return classIndex->get(h);
}

vector<int> FeatureIndexesPair::getWordFeatures(int class_id, Hash h) const {
  return wordIndexes[class_id]->get(h);
}

void FeatureIndexesPair::addClassIndex(Hash h, int class_id) {
  classIndex->add(h, class_id);
}

void FeatureIndexesPair::addWordIndex(int class_id, Hash h, int word_class_id) {
  wordIndexes[class_id]->add(h, word_class_id);
}

} // namespace oxlm
