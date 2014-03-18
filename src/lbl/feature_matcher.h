#pragma once

#include <boost/shared_ptr.hpp>

#include "lbl/context_extractor.h"
#include "lbl/feature_context.h"
#include "lbl/feature_generator.h"
#include "lbl/word_to_class_index.h"

using namespace std;

namespace oxlm {

class FeatureMatcher {
 public:
  FeatureMatcher(
      const Corpus& corpus, const WordToClassIndex& index,
      const ContextExtractor& extractor,
      const boost::shared_ptr<FeatureGenerator>& generator);

  MatchingContexts getClassFeatures() const;

  MatchingContexts getWordFeatures(int class_id) const;

 private:
  MatchingContexts class_features;
  vector<MatchingContexts> word_features;
};

} // namespace oxlm
