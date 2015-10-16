#include "lbl/feature_filter.h"

namespace oxlm {

FeatureFilter::FeatureFilter() {}

FeatureFilter::FeatureFilter(const FeatureIndexesPtr& feature_indexes)
    : featureIndexes(feature_indexes) {}

vector<int> FeatureFilter::getIndexes(Hash context_hash) const {
  auto it = featureIndexes->find(context_hash);
  if (it == featureIndexes->end()) {
    return vector<int>();
  }

  return it->second;
}

bool FeatureFilter::hasIndex(Hash context_hash, int feature_index) const {
  auto it = featureIndexes->find(context_hash);
  if (it == featureIndexes->end()) {
    return false;
  }

  const auto& indexes = it->second;
  return find(indexes.begin(), indexes.end(), feature_index) != indexes.end();
}

FeatureFilter::~FeatureFilter() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::FeatureFilter)
