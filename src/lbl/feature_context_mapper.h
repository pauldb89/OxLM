#pragma once

#include <vector>

#include <boost/serialization/serialization.hpp>

#include "lbl/context_processor.h"
#include "lbl/feature_context.h"
#include "lbl/feature_context_generator.h"
#include "lbl/ngram_filter.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

/**
 * Given a context of words [w_{n-1}, w_{n-2}, ...] generates all the feature
 * ids that match the context.
 *
 * Preprocesses the training corpus to populate the hash table with all the
 * possible feature ids to guarantee thread-safe operations later on.
 **/
class FeatureContextMapper {
 public:
  FeatureContextMapper();

  FeatureContextMapper(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<ContextProcessor>& processor,
      const boost::shared_ptr<FeatureContextGenerator>& generator,
      const boost::shared_ptr<NGramFilter>& filter);

  // Unobserved contexts are skipped.
  vector<int> getClassContextIds(const vector<int>& context) const;

  vector<int> getClassContextIds(
      const vector<FeatureContext>& feature_contexts) const;

  vector<int> getWordContextIds(int class_id, const vector<int>& context) const;

  vector<int> getWordContextIds(
      int class_id, const vector<FeatureContext>& feature_contexts) const;

  int getClassContextId(const FeatureContext& feature_context) const;

  int getWordContextId(int class_id, const FeatureContext& feature_context) const;

  int getNumContexts() const;

  int getNumClassContexts() const;

  int getNumWordContexts(int class_id) const;

  bool operator==(const FeatureContextMapper& other) const;

private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & index;
    ar & generator;
    ar & classContextIdsMap;
    ar & wordContextIdsMap;
  }

  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FeatureContextGenerator> generator;
  hash<FeatureContext> hashFunction;
  unordered_map<size_t, int> classContextIdsMap;
  vector<unordered_map<size_t, int>> wordContextIdsMap;
};

} // namespace oxlm
