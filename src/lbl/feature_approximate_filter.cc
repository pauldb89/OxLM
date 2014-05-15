#include "lbl/feature_approximate_filter.h"

namespace oxlm {

FeatureApproximateFilter::FeatureApproximateFilter() {}

FeatureApproximateFilter::FeatureApproximateFilter(
    int num_candidates, const boost::shared_ptr<FeatureContextKeyer>& keyer,
    const boost::shared_ptr<BloomFilter<NGramQuery>>& bloom_filter)
    : numCandidates(num_candidates), keyer(keyer), bloomFilter(bloom_filter) {}

vector<int> FeatureApproximateFilter::getIndexes(
    const FeatureContext& feature_context) const {
  vector<int> indexes;
  for (int index = 0; index < numCandidates; ++index) {
    if (bloomFilter->contains(keyer->getPrediction(index, feature_context))) {
      indexes.push_back(index);
    }
  }
  return indexes;
}

bool FeatureApproximateFilter::operator==(const FeatureApproximateFilter& other) const {
  return numCandidates == other.numCandidates && *bloomFilter == *other.bloomFilter;
}

FeatureApproximateFilter::~FeatureApproximateFilter() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::FeatureApproximateFilter)
