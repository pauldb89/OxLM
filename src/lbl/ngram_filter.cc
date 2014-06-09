#include "lbl/ngram_filter.h"

namespace oxlm {

NGramFilter::NGramFilter(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextGenerator>& generator,
    int max_ngrams, int min_ngram_freq) {
  enabled = max_ngrams > 0 || min_ngram_freq > 1;
  if (!enabled) {
    return;
  }

  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    vector<int> context = processor->extract(i);
    for (const auto& feature_context: generator->getFeatureContexts(context)) {
      NGram ngram(word_id, class_id, feature_context.data);
      size_t ngram_hash = hashFunction(ngram);
      ++ngramFrequencies[ngram_hash];
    }
  }

  cout << "Total number of n-grams: " << ngramFrequencies.size()
       << "..." << endl;

  vector<pair<int, size_t>> ngrams;
  for (const auto& ngram_frequency: ngramFrequencies) {
    if (ngram_frequency.second >= min_ngram_freq) {
      ngrams.push_back(make_pair(
          ngram_frequency.second, ngram_frequency.first));
    }
  }

  cout << "Number of n-grams above minimum frequency threshold: "
       << ngrams.size() << endl;

  if (max_ngrams != 0 && ngrams.size() > max_ngrams) {
    partial_sort(ngrams.begin(), ngrams.begin() + max_ngrams, ngrams.end(),
        greater<pair<int, size_t>>());
    ngrams.resize(max_ngrams);
    cout << "n-gram minimum frequency " << ngrams.back().first << "..." << endl;
  } else {
    cout << "n-gram minimum frequency " << min_ngram_freq << endl;
  }


  ngramFrequencies.clear();
  for (const auto& ngram: ngrams) {
    ngramFrequencies[ngram.second] = ngram.first;
  }
}

vector<FeatureContext> NGramFilter::filter(
    int word_id, int class_id,
    const vector<FeatureContext>& feature_contexts) const {
  if (!enabled) {
    return feature_contexts;
  }

  vector<FeatureContext> ret;
  for (const auto& feature_context: feature_contexts) {
    NGram ngram(word_id, class_id, feature_context.data);
    size_t ngram_hash = hashFunction(ngram);
    if (ngramFrequencies.count(ngram_hash)) {
      ret.push_back(feature_context);
    }
  }

  return ret;
}

} // namespace oxlm
