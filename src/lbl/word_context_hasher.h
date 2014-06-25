#pragma once

#include "lbl/archive_export.h"
#include "lbl/feature_context_hasher.h"
#include "lbl/ngram.h"

namespace oxlm {

class WordContextHasher : public FeatureContextHasher {
 public:
  WordContextHasher();

  WordContextHasher(int class_id, int hash_space_size);

  virtual int getKey(const FeatureContext& feature_context) const;

  virtual NGram getPrediction(
      int candidate, const FeatureContext& feature_context) const;

  bool operator==(const WordContextHasher& other) const;

  virtual ~WordContextHasher();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextHasher>(*this);
    ar & classId;
    ar & hashSpaceSize;
  }

  int classId, hashSpaceSize;
  hash<NGram> hash_function;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::WordContextHasher)
