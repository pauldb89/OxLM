#include "lbl/bloom_filter_populator.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_hasher.h"
#include "lbl/context_processor.h"
#include "lbl/feature_context_generator.h"
#include "lbl/word_context_hasher.h"

namespace oxlm {

BloomFilterPopulator::BloomFilterPopulator() {}

BloomFilterPopulator::BloomFilterPopulator(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextMapper>& mapper,
    const boost::shared_ptr<ModelData>& config) {
  // This is only an underestimate of the number of distinct n-grams in the
  // training data (feature contexts do not include the predict word) =>
  // #(feature_contexts) < #(ngrams).
  bloomFilter = boost::make_shared<BloomFilter<NGram>>(
      mapper->getNumContexts(), 1, config->filter_error_rate);

  ContextProcessor processor(corpus, config->ngram_order - 1);
  FeatureContextGenerator generator(config->feature_context_size);
  ClassContextHasher class_hasher(config->hash_space);
  vector<WordContextHasher> word_hashers;
  for (int i = 0; i < index->getNumClasses(); ++i) {
    word_hashers.push_back(WordContextHasher(i, config->hash_space));
  }

  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    vector<int> context = processor.extract(i);
    for (const auto& feature_context: generator.getFeatureContexts(context)) {
      bloomFilter->increment(
          class_hasher.getPrediction(class_id, feature_context));
      bloomFilter->increment(
          word_hashers[class_id].getPrediction(word_class_id, feature_context));
    }
  }
}

boost::shared_ptr<BloomFilter<NGram>> BloomFilterPopulator::get() const {
  return bloomFilter;
}

bool BloomFilterPopulator::operator==(const BloomFilterPopulator& other) const {
  return *bloomFilter == *other.bloomFilter;
}

} // namespace oxlm
