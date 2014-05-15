#pragma once

#include "lbl/archive_export.h"
#include "lbl/feature_context_keyer.h"
#include "lbl/ngram_query.h"

namespace oxlm {

class WordContextKeyer : public FeatureContextKeyer {
 public:
  WordContextKeyer();

  WordContextKeyer(int class_id, int num_words, int hash_space_size);

  virtual int getKey(const FeatureContext& feature_context) const;

  virtual NGramQuery getPrediction(
      int candidate, const FeatureContext& feature_context) const;

  bool operator==(const WordContextKeyer& other) const;

  virtual ~WordContextKeyer();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextKeyer>(*this);
    ar & classId;
    ar & numWords;
    ar & hashSpaceSize;
  }

  int classId, numWords, hashSpaceSize;
  hash<NGramQuery> hash_function;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::WordContextKeyer)
