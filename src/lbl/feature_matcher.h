#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/context_processor.h"
#include "lbl/feature_context.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/word_to_class_index.h"

using namespace std;

namespace oxlm {

class FeatureMatcher {
 public:
  FeatureMatcher(
      const Corpus& corpus, const WordToClassIndex& index,
      const ContextProcessor& processor,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  FeatureIndexes getClassFeatures() const;

  FeatureIndexes getWordFeatures(int class_id) const;

 private:
  const WordToClassIndex& index;
  const ContextProcessor& processor;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  FeatureIndexes class_features;
  vector<FeatureIndexes> word_features;
};

} // namespace oxlm
