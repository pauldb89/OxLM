#pragma once

#include <boost/serialization/serialization.hpp>

#include "lbl/feature_context.h"

namespace oxlm {

class FeatureContextKeyer {
 public:
  virtual int getKey(const FeatureContext& feature_context) const = 0;

  virtual ~FeatureContextKeyer();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {}
};

} // namespace oxlm
