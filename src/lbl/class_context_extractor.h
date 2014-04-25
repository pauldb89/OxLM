#pragma once

// We need to include the archives in the shared library so BOOST_EXPORT knows
// to register implementations for all archives in use.
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/feature_context_extractor.h"
#include "lbl/feature_context_hasher.h"

namespace oxlm {

class ClassContextExtractor : public FeatureContextExtractor {
 public:
  ClassContextExtractor();

  ClassContextExtractor(const boost::shared_ptr<FeatureContextHasher>& hasher);

  virtual vector<int> getFeatureContextIds(const vector<int>& context) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextExtractor>(*this);
    ar & hasher;
  }

  boost::shared_ptr<FeatureContextHasher> hasher;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::ClassContextExtractor)
