#include "lbl/ngram_filter.h"

namespace oxlm {

NGramFilter::NGramFilter(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextGenerator>& generator,
    int max_ngrams)
    : maxNGrams(max_ngrams) {
  if (maxNGrams == 0) {
    return;
  }

  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    vector<int> context = processor->extract(i);
    for (const auto& feature_context: generator->getFeatureContexts(context)) {
      NGramQuery ngram(word_id, class_id, feature_context.data);
      size_t ngram_hash = hashFunction(ngram);
      ++ngramFrequencies[ngram_hash];
    }
  }

  vector<pair<int, size_t>> ngrams;
  for (const auto& ngram_frequency: ngramFrequencies) {
    ngrams.push_back(make_pair(ngram_frequency.second, ngram_frequency.first));
  }

  if (ngrams.size() > max_ngrams) {
    partial_sort(ngrams.begin(), ngrams.begin() + max_ngrams, ngrams.end(),
        greater<pair<int, size_t>>());
    ngrams.resize(max_ngrams);
  }

  cout << "n-gram minimum frequency " << ngrams.back().first << "..." << endl;

  ngramFrequencies.clear();
  for (const auto& ngram: ngrams) {
    ngramFrequencies[ngram.second] = ngram.first;
  }
}

vector<FeatureContext> NGramFilter::filter(
    int word_id, int class_id,
    const vector<FeatureContext>& feature_contexts) const {
  if (maxNGrams == 0) {
    return feature_contexts;
  }

  vector<FeatureContext> ret;
  for (const auto& feature_context: feature_contexts) {
    NGramQuery ngram(word_id, class_id, feature_context.data);
    size_t ngram_hash = hashFunction(ngram);
    if (ngramFrequencies.count(ngram_hash)) {
      ret.push_back(feature_context);
    }
  }

  return ret;
}

} // namespace oxlm
