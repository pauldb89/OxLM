#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/context_processor.h"
#include "lbl/feature_context.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/global_feature_indexes_pair.h"
#include "lbl/minibatch_feature_indexes_pair.h"
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

  GlobalFeatureIndexesPairPtr getGlobalFeatures() const;

  MinibatchFeatureIndexesPairPtr getMinibatchFeatures(
      const vector<int>& minibatch_indexes) const;

 private:
  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<ContextProcessor> processor;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  GlobalFeatureIndexesPairPtr feature_indexes;
};

} // namespace oxlm
