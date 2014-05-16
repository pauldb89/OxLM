#pragma once

#include "lbl/bloom_filter.h"
#include "lbl/config.h"
#include "lbl/feature_context_hasher.h"
#include "lbl/ngram_query.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class BloomFilterPopulator {
 public:
  BloomFilterPopulator(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextHasher>& hasher,
      const ModelData& config);

  boost::shared_ptr<BloomFilter<NGramQuery>> get() const;

 private:
  boost::shared_ptr<BloomFilter<NGramQuery>> bloomFilter;
};

} // namespace oxlm
