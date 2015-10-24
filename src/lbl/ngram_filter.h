#pragma once

#include <unordered_set>

#include "lbl/context_processor.h"
#include "lbl/feature_context_generator.h"
#include "lbl/hashed_ngram.h"
#include "lbl/ngram.h"
#include "lbl/word_to_class_index.h"
#include "lbl/parallel_corpus.h"

namespace oxlm {

class NGramFilter {
 public:
  NGramFilter(const string& ngram_file);

  NGramFilter(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<ContextProcessor>& processor,
      const boost::shared_ptr<FeatureContextGenerator>& generator,
      int max_ngrams = 0, int min_ngram_freq = 1);

  vector<Hash> filter(
      int word_id, int class_id, const vector<Hash>& context_hashes) const;

 private:
  bool enabled;
  hash<HashedNGram> hasher;
  unordered_map<size_t, int> ngramFrequencies;
  unordered_set<size_t> validNGrams;
};

} // namespace oxlm
