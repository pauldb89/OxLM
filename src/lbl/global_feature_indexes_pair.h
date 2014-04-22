#pragma once

#include <vector>

#include "lbl/feature_context_extractor.h"
#include "lbl/word_to_class_index.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class GlobalFeatureIndexesPair {
 public:
  GlobalFeatureIndexesPair(
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  GlobalFeatureIndexesPtr getClassIndexes() const;

  GlobalFeatureIndexesPtr getWordIndexes(int class_id) const;

  vector<int> getClassFeatures(int feature_context_id) const;

  vector<int> getWordFeatures(
      int class_id, int feature_context_id) const;

  void addClassIndex(int feature_context_id, int class_id);

  void addWordIndex(
      int class_id, int feature_context_id, int word_class_id);

 private:
  GlobalFeatureIndexesPtr class_indexes;
  vector<GlobalFeatureIndexesPtr> word_indexes;
};

typedef boost::shared_ptr<GlobalFeatureIndexesPair> GlobalFeatureIndexesPairPtr;

} // namespace oxlm
