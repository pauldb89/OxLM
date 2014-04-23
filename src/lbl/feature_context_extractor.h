#pragma once

#include <vector>

#include <boost/serialization/serialization.hpp>

#include "lbl/context_processor.h"
#include "lbl/feature_context.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

typedef unordered_map<FeatureContext, int, hash<FeatureContext>>
    FeatureContextHash;

/**
 * Given a context of words [w_{n-1}, w_{n-2}, ...] generates all the feature
 * ids that match the context.
 *
 * Preprocesses the training corpus to populate the hash table with all the
 * possible feature ids to guarantee thread-safe operations later on.
 **/
class FeatureContextExtractor {
 public:
  FeatureContextExtractor();

  FeatureContextExtractor(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<ContextProcessor>& processor,
      size_t feature_context_size);

  // Produces a pair of class context ids and word context ids given the class.
  // Unobserved contexts are skipped.
  pair<vector<int>, vector<int>> getFeatureContextIds(
      int class_id, const vector<WordId>& history) const;

  int getNumClassContexts() const;

  int getNumWordContexts(int class_id) const;

private:
  vector<FeatureContext> getFeatureContexts(
      const vector<WordId>& history) const;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & index;
    ar & featureContextSize;
    ar & classContextIdsMap;
    ar & wordContextIdsMap;
  }

  boost::shared_ptr<WordToClassIndex> index;

  size_t featureContextSize;
  FeatureContextHash classContextIdsMap;
  vector<FeatureContextHash> wordContextIdsMap;
};

} // namespace oxlm
