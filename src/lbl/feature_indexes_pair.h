#pragma once

#include <vector>

#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class FeatureIndexesPair {
 public:
  FeatureIndexesPair(int num_classes);

  FeatureIndexesPtr getClassIndexes() const;

  FeatureIndexesPtr getWordIndexes(int class_id) const;

  unordered_set<int> getClassFeatures(int feature_context_id) const;

  unordered_set<int> getWordFeatures(
      int class_id, int feature_context_id) const;

  void addClassIndex(FeatureContextId feature_context_id, int class_id);

  void addWordIndex(
      int class_id, FeatureContextId feature_context_id, int word_class_id);

  void setClassIndexes(
      FeatureContextId feature_context_id,
      const unordered_set<int>& indexes);

  void setWordIndexes(
      int class_id,
      FeatureContextId feature_context_id,
      const unordered_set<int>& indexes);

 private:
  FeatureIndexesPtr class_indexes;
  vector<FeatureIndexesPtr> word_indexes;
};

typedef boost::shared_ptr<FeatureIndexesPair> FeatureIndexesPairPtr;

} // namespace oxlm
