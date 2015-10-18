#pragma once

#include "lbl/archive_export.h"
#include "lbl/feature_context_hasher.h"
#include "lbl/hashed_ngram.h"

namespace oxlm {

class WordContextHasher : public FeatureContextHasher {
 public:
  WordContextHasher();

  WordContextHasher(int class_id);

  virtual size_t getKey(Hash context_hash) const;

  bool operator==(const WordContextHasher& other) const;

  virtual ~WordContextHasher();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextHasher>(*this);
    ar & classId;
  }

  int classId;
  hash<HashedNGram> hashFunction;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::WordContextHasher)
