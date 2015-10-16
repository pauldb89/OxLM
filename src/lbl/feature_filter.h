#pragma once

#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/type_info_implementation.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_filter.h"
#include "lbl/utils.h"
#include "utils/serialization_helpers.h"

namespace oxlm {

class FeatureFilter {
 public:
  FeatureFilter();

  FeatureFilter(const FeatureIndexesPtr& feature_indexes);

  virtual vector<int> getIndexes(Hash context_hash) const;

  virtual bool hasIndex(Hash context_hash, int feature_index) const;

  ~FeatureFilter();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & featureIndexes;
  }

  FeatureIndexesPtr featureIndexes;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::FeatureFilter)
