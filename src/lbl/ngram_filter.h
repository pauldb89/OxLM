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
  NGramFilter();

  NGramFilter(const string& ngram_file);

  NGramFilter(const unordered_set<size_t>& valid_ngrams);

  vector<Hash> filter(
      int word_id, int class_id, const vector<Hash>& context_hashes) const;

 private:
  bool enabled;
  hash<HashedNGram> hasher;
  unordered_set<size_t> validNGrams;
};

} // namespace oxlm
