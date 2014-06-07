#pragma once

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/feature_filter.h"
#include "lbl/utils.h"

namespace oxlm {

class FeatureExactFilter : public FeatureFilter {
 public:
  FeatureExactFilter();

  FeatureExactFilter(
      const GlobalFeatureIndexesPtr& feature_indexes,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  virtual vector<int> getIndexes(const FeatureContext& feature_context) const;

  ~FeatureExactFilter();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureFilter>(*this);
    ar & featureIndexes;
    ar & extractor;
  }

  GlobalFeatureIndexesPtr featureIndexes;
  boost::shared_ptr<FeatureContextExtractor> extractor;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::FeatureExactFilter)
