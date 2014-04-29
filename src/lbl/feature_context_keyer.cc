#include "lbl/feature_context_keyer.h"

namespace oxlm {

FeatureContextKeyer::FeatureContextKeyer() {}

FeatureContextKeyer::FeatureContextKeyer(
    int hash_space, int feature_context_size)
    : hashSpace(hash_space), generator(feature_context_size) {}

vector<int> FeatureContextKeyer::getKeys(const vector<int>& context) const {
  vector<int> keys;
  for (const auto& feature_context: generator.getFeatureContexts(context)) {
    keys.push_back(hash_function(feature_context) % hashSpace);
  }

  return keys;
}

bool FeatureContextKeyer::operator==(const FeatureContextKeyer& other) const {
  return generator == other.generator;
}

} // namespace oxlm
