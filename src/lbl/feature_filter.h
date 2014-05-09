#pragma once

#include <vector>

#include <boost/serialization/serialization.hpp>

#include "lbl/feature_context.h"

using namespace std;

namespace oxlm {

/**
 * @Interface.
 * Feature filters are used to identify what words may follow a given context.
 */
class FeatureFilter {
 public:
  virtual vector<int> getIndexes(
      const FeatureContext& feature_context) const = 0;

  virtual ~FeatureFilter();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {}
};

} // namespace oxlm
