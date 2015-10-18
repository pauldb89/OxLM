#pragma once

#include <boost/serialization/serialization.hpp>

#include "lbl/feature_context.h"
#include "lbl/ngram.h"

namespace oxlm {

class FeatureContextHasher {
 public:
  virtual size_t getKey(Hash context_hash) const = 0;

  virtual ~FeatureContextHasher();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {}
};

} // namespace oxlm
