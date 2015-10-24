#include "lbl/ngram_filter.h"

namespace oxlm {

NGramFilter::NGramFilter(const string& ngram_file) {
  ifstream fin(ngram_file);
  size_t ngram_hash;
  while (fin >> ngram_hash) {
    validNGrams.insert(ngram_hash);
  }

  enabled = validNGrams.size() > 0;
  cerr << "Read " << validNGrams.size() << " n-grams..." << endl;
}

NGramFilter::NGramFilter() : enabled(false) {}

NGramFilter::NGramFilter(const unordered_set<size_t>& valid_ngrams)
  : enabled(valid_ngrams.size() > 0), validNGrams(valid_ngrams) {}

vector<Hash> NGramFilter::filter(
    int word_id, int class_id, const vector<Hash>& context_hashes) const {
  if (!enabled) {
    return context_hashes;
  }

  vector<Hash> ret;
  for (Hash context_hash: context_hashes) {
    HashedNGram ngram(word_id, class_id, context_hash);
    size_t ngram_hash = hasher(ngram);
    if (validNGrams.count(ngram_hash)) {
      ret.push_back(context_hash);
    }
  }

  return ret;
}

} // namespace oxlm
