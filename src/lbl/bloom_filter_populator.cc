#include "lbl/bloom_filter_populator.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_keyer.h"
#include "lbl/context_processor.h"
#include "lbl/feature_context_generator.h"
#include "lbl/word_context_keyer.h"

namespace oxlm {

BloomFilterPopulator::BloomFilterPopulator(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const ModelData& config) {
  // This is only an underestimate of the number of distinct n-grams in the
  // training data (feature contexts do not include the predict word) =>
  // #(feature_contexts) < #(ngrams).
  bloomFilter = boost::make_shared<BloomFilter<NGramQuery>>(
      hasher->getNumContexts(), 1, config.filter_error_rate);

  ContextProcessor processor(corpus, config.ngram_order - 1);
  FeatureContextGenerator generator(config.feature_context_size);
  ClassContextKeyer class_keyer(config.hash_space);
  vector<WordContextKeyer> word_keyers;
  for (int i = 0; i < index->getNumClasses(); ++i) {
    word_keyers.push_back(WordContextKeyer(
        i, index->getNumWords(), config.hash_space));
  }

  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    vector<int> context = processor.extract(i);
    for (const auto& feature_context: generator.getFeatureContexts(context)) {
      bloomFilter->increment(
          class_keyer.getPrediction(class_id, feature_context));
      bloomFilter->increment(
          word_keyers[class_id].getPrediction(word_class_id, feature_context));
    }
  }
}

boost::shared_ptr<BloomFilter<NGramQuery>> BloomFilterPopulator::get() const {
  return bloomFilter;
}

} // namespace oxlm
