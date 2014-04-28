#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

#include "lbl/utils.h"
#include "lbl/feature_context.h"

using namespace std;

namespace oxlm {

class FeatureStore {
 public:
  virtual VectorReal get(const vector<int>& context) const = 0;

  virtual size_t size() const = 0;

  virtual ~FeatureStore();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {}
};

} // namespace oxlm
