#include "lbl/feature_no_op_filter.h"

namespace oxlm {

FeatureNoOpFilter::FeatureNoOpFilter() {}

FeatureNoOpFilter::FeatureNoOpFilter(int group_size) : groupSize(group_size) {}

vector<int> FeatureNoOpFilter::getIndexes(
    const FeatureContext& feature_context) const {
  vector<int> indexes(groupSize);
  iota(indexes.begin(), indexes.end(), 0);
  return indexes;
}

bool FeatureNoOpFilter::operator==(const FeatureNoOpFilter& other) const {
  return groupSize == other.groupSize;
}

FeatureNoOpFilter::~FeatureNoOpFilter() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::FeatureNoOpFilter)
