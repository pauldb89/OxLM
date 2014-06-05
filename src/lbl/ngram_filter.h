#pragma once

#include "lbl/context_processor.h"
#include "lbl/feature_context_generator.h"
#include "lbl/ngram_query.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class NGramFilter {
 public:
  NGramFilter(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<ContextProcessor>& processor,
      const boost::shared_ptr<FeatureContextGenerator>& generator,
      int max_ngrams = 0);

  vector<FeatureContext> filter(
      int word_id, int class_id,
      const vector<FeatureContext>& feature_contexts) const;

 private:
  int maxNGrams;
  hash<NGramQuery> hashFunction;
  unordered_map<size_t, int> ngramFrequencies;
};

} // namespace oxlm
