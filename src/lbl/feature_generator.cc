#include "feature_generator.h"

namespace oxlm {

FeatureGenerator::FeatureGenerator(size_t feature_context_size) :
    feature_context_size(feature_context_size) {}

vector<FeatureContext> FeatureGenerator::generate(
    const vector<int>& history) const {
  vector<FeatureContext> feature_contexts;
  vector<int> context;
  for (size_t i = 0; i < min(feature_context_size, history.size()); ++i) {
    context.push_back(history[i]);
    feature_contexts.push_back(FeatureContext(i, context));
  }
  return feature_contexts;
}

} // namespace oxlm
