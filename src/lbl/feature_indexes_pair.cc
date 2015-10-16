#include "lbl/feature_indexes_pair.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureIndexesPair::FeatureIndexesPair() {}

FeatureIndexesPair::FeatureIndexesPair(const boost::shared_ptr<WordToClassIndex>& index) {
  classIndexes = boost::make_shared<FeatureIndexes>();
  for (int i = 0; i < index->getNumClasses(); ++i) {
    wordIndexes.push_back(boost::make_shared<FeatureIndexes>());
  }
}

FeatureIndexesPtr FeatureIndexesPair::getClassIndexes() const {
  return classIndexes;
}

FeatureIndexesPtr FeatureIndexesPair::getWordIndexes(int class_id) const {
  return wordIndexes[class_id];
}

vector<int> FeatureIndexesPair::getClassFeatures(Hash h) const {
  return classIndexes->at(h);
}

vector<int> FeatureIndexesPair::getWordFeatures(int class_id, Hash h) const {
  return wordIndexes[class_id]->at(h);
}

void FeatureIndexesPair::addClassIndex(Hash h, int class_id) {
  vector<int>& indexes = classIndexes->operator[](h);
  if (find(indexes.begin(), indexes.end(), class_id) == indexes.end()) {
    indexes.reserve(indexes.size() + 1);
    indexes.push_back(class_id);
  }
}

void FeatureIndexesPair::addWordIndex(int class_id, Hash h, int word_class_id) {
  vector<int>& indexes = wordIndexes[class_id]->operator[](h);
  if (find(indexes.begin(), indexes.end(), word_class_id) == indexes.end()) {
    indexes.reserve(indexes.size() + 1);
    indexes.push_back(word_class_id);
  }
}

} // namespace oxlm
