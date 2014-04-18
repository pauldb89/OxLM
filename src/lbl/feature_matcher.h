#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/context_processor.h"
#include "lbl/feature_context.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/feature_indexes_pair.h"
#include "lbl/word_to_class_index.h"

using namespace std;

namespace oxlm {

class FeatureMatcher {
 public:
  FeatureMatcher(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<ContextProcessor>& processor,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  FeatureIndexesPairPtr getFeatures() const;

  FeatureIndexesPairPtr getFeatures(const vector<int>& minibatch_indexes) const;

 private:
  void addFeatureIndexes(
      int training_index, FeatureIndexesPairPtr feature_indexes) const;

  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<ContextProcessor> processor;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  FeatureIndexesPairPtr feature_indexes;
};

} // namespace oxlm
