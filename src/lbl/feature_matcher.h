#pragma once

#include "lbl/context_extractor.h"
#include "lbl/feature_context.h"
#include "lbl/feature_generator.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FeatureMatcher {
 public:
  FeatureMatcher(
      const Corpus& corpus, const WordToClassIndex& index,
      const ContextExtractor& extractor, const FeatureGenerator& generator);

  MatchingContexts getClassFeatures() const;

  MatchingContexts getWordFeatures(int class_id) const;

 private:
  MatchingContexts class_features;
  vector<MatchingContexts> word_features;
};

} // namespace oxlm
