#pragma once

#include <boost/serialization/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/feature_context_hasher.h"

namespace oxlm {

class ClassContextExtractor : public FeatureContextExtractor {
 public:
  ClassContextExtractor();

  ClassContextExtractor(const boost::shared_ptr<FeatureContextHasher>& hasher);

  virtual vector<int> getFeatureContextIds(const vector<int>& context) const;

  bool operator==(const ClassContextExtractor& extractor) const;

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
