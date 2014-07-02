#pragma once

#include "lbl/bloom_filter.h"
#include "lbl/config.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/ngram.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

#include <boost/serialization/shared_ptr.hpp>

namespace oxlm {

class BloomFilterPopulator {
 public:
  BloomFilterPopulator();

  BloomFilterPopulator(
      const boost::shared_ptr<Corpus>& corpus,
      const boost::shared_ptr<WordToClassIndex>& index,
      const boost::shared_ptr<FeatureContextMapper>& mapper,
      const boost::shared_ptr<ModelData>& config);

  boost::shared_ptr<BloomFilter<NGram>> get() const;

  bool operator==(const BloomFilterPopulator& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & bloomFilter;
  }

  boost::shared_ptr<BloomFilter<NGram>> bloomFilter;
};

} // namespace oxlm
