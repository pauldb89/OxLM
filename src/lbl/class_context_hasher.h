#pragma once

#include "lbl/archive_export.h"
#include "lbl/feature_context_hasher.h"

namespace oxlm {

class ClassContextHasher : public FeatureContextHasher {
 public:
  ClassContextHasher();

  virtual size_t getKey(Hash context_hash) const;

  bool operator==(const ClassContextHasher& other) const;

  virtual ~ClassContextHasher();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextHasher>(*this);
  }
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::ClassContextHasher)
