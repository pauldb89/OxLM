#pragma once

#include <unordered_set>

#include "lbl/config.h"
#include "lbl/feature_context.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class CollisionCounter {
 public:
  CollisionCounter(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const ModelData& config);

  int count() const;

 private:
  ModelData config;
  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  unordered_set<FeatureContext> observedClassContexts;
  vector<unordered_set<FeatureContext>> observedWordContexts;
  unordered_set<int> observedClassKeys;
  vector<unordered_set<int>> observedWordKeys;
};

} // namespace oxlm
