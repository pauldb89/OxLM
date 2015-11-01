#include "lbl/feature_filter.h"

namespace oxlm {

FeatureFilter::FeatureFilter() {}

FeatureFilter::FeatureFilter(const FeatureIndexPtr& feature_indexes)
    : featureIndex(feature_indexes) {}

vector<int> FeatureFilter::getIndexes(Hash context_hash) const {
  return featureIndex->get(context_hash);
}

bool FeatureFilter::hasIndex(Hash context_hash, int feature_index) const {
  return featureIndex->contains(context_hash, feature_index);
}

FeatureFilter::~FeatureFilter() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::FeatureFilter)
