#pragma once

#include <vector>

#include "lbl/word_to_class_index.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class FeatureIndexesPair {
 public:
  FeatureIndexesPair();

  FeatureIndexesPair(const boost::shared_ptr<WordToClassIndex>& index);

  FeatureIndexesPtr getClassIndexes() const;

  FeatureIndexesPtr getWordIndexes(int class_id) const;

  vector<int> getClassFeatures(Hash h) const;

  vector<int> getWordFeatures(int class_id, Hash h) const;

  void addClassIndex(Hash h, int class_id);

  void addWordIndex(int class_id, Hash h, int word_class_id);

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & classIndexes;
    ar & wordIndexes;
  }

  FeatureIndexesPtr classIndexes;
  vector<FeatureIndexesPtr> wordIndexes;
};

typedef boost::shared_ptr<FeatureIndexesPair> FeatureIndexesPairPtr;

} // namespace oxlm
