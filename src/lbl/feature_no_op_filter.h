#pragma once

#include "lbl/archive_export.h"
#include "lbl/feature_filter.h"

namespace oxlm {

class FeatureNoOpFilter : public FeatureFilter {
 public:
  FeatureNoOpFilter();

  FeatureNoOpFilter(int group_size);

  virtual vector<int> getIndexes(const FeatureContext& feature_context) const;

  bool operator==(const FeatureNoOpFilter& other) const;

  virtual ~FeatureNoOpFilter();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureFilter>(*this);
    ar & groupSize;
  }

  int groupSize;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::FeatureNoOpFilter)
