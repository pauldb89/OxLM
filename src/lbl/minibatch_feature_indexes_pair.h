#pragma once

#include <vector>

#include "lbl/feature_context_extractor.h"
#include "lbl/word_to_class_index.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class MinibatchFeatureIndexesPair {
 public:
  MinibatchFeatureIndexesPair(
      const boost::shared_ptr<WordToClassIndex>& index);

  MinibatchFeatureIndexesPtr getClassIndexes() const;

  MinibatchFeatureIndexesPtr getWordIndexes(int class_id) const;

  void setClassIndexes(int feature_context_id, const vector<int>& indexes);

  void setWordIndexes(
      int class_id,
      int feature_context_id,
      const vector<int>& indexes);

 private:
  MinibatchFeatureIndexesPtr class_indexes;
  vector<MinibatchFeatureIndexesPtr> word_indexes;
};

typedef boost::shared_ptr<MinibatchFeatureIndexesPair>
    MinibatchFeatureIndexesPairPtr;

} // namespace oxlm
