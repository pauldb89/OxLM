#pragma once

#include <vector>

#include "lbl/feature_context_mapper.h"
#include "lbl/word_to_class_index.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class GlobalFeatureIndexesPair {
 public:
  GlobalFeatureIndexesPair();

  GlobalFeatureIndexesPair(
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextMapper>& mapper);

  GlobalFeatureIndexesPtr getClassIndexes() const;

  GlobalFeatureIndexesPtr getWordIndexes(int class_id) const;

  vector<int> getClassFeatures(int feature_context_id) const;

  vector<int> getWordFeatures(
      int class_id, int feature_context_id) const;

  void addClassIndex(int feature_context_id, int class_id);

  void addWordIndex(
      int class_id, int feature_context_id, int word_class_id);

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & classIndexes;
    ar & wordIndexes;
  }

  GlobalFeatureIndexesPtr classIndexes;
  vector<GlobalFeatureIndexesPtr> wordIndexes;
};

typedef boost::shared_ptr<GlobalFeatureIndexesPair> GlobalFeatureIndexesPairPtr;

} // namespace oxlm
