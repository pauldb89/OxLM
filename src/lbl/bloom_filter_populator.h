#pragma once

#include "lbl/bloom_filter.h"
#include "lbl/config.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/ngram.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class BloomFilterPopulator {
 public:
  BloomFilterPopulator(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextMapper>& mapper,
      const ModelData& config);

  boost::shared_ptr<BloomFilter<NGram>> get() const;

 private:
  boost::shared_ptr<BloomFilter<NGram>> bloomFilter;
};

} // namespace oxlm
