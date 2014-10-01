#include "lbl/feature_exact_filter.h"

namespace oxlm {

FeatureExactFilter::FeatureExactFilter() {}

FeatureExactFilter::FeatureExactFilter(
    const GlobalFeatureIndexesPtr& feature_indexes,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : featureIndexes(feature_indexes), extractor(extractor) {}

vector<int> FeatureExactFilter::getIndexes(
    const FeatureContext& feature_context) const {
  int feature_context_id = extractor->getFeatureContextId(feature_context);
  if (feature_context_id == -1) {
    return vector<int>();
  } else {
    return featureIndexes->at(feature_context_id);
  }
}

bool FeatureExactFilter::hasIndex(
    const FeatureContext& feature_context, int feature_index) const {
  int feature_context_id = extractor->getFeatureContextId(feature_context);
  if (feature_context_id == -1) {
    return false;
  }

  const auto& indexes = featureIndexes->at(feature_context_id);
  return find(indexes.begin(), indexes.end(), feature_index) != indexes.end();
}

FeatureExactFilter::~FeatureExactFilter() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::FeatureExactFilter)
