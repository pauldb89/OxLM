#include "lbl/feature_context_keyer.h"

namespace oxlm {

FeatureContextKeyer::FeatureContextKeyer() {}

FeatureContextKeyer::FeatureContextKeyer(int feature_context_size)
    : generator(feature_context_size) {}

vector<size_t> FeatureContextKeyer::getKeys(const vector<int>& context) const {
  vector<size_t> keys;
  for (const auto& feature_context: generator.getFeatureContexts(context)) {
    keys.push_back(hash_function(feature_context));
  }

  return keys;
}

bool FeatureContextKeyer::operator==(const FeatureContextKeyer& other) const {
  return generator == other.generator;
}

} // namespace oxlm
