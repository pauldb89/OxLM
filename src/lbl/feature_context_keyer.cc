#include "lbl/feature_context_keyer.h"

namespace oxlm {

FeatureContextKeyer::FeatureContextKeyer() {}

FeatureContextKeyer::FeatureContextKeyer(int hash_space)
    : hashSpace(hash_space) {}

int FeatureContextKeyer::getKey(const FeatureContext& feature_context) const {
  return hash_function(feature_context) % hashSpace;
}

bool FeatureContextKeyer::operator==(const FeatureContextKeyer& other) const {
  return hashSpace == other.hashSpace;
}

} // namespace oxlm
