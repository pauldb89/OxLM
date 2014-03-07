#include "feature_generator.h"

namespace oxlm {

FeatureGenerator::FeatureGenerator(size_t feature_context_size) :
    feature_context_size(feature_context_size) {}

vector<Feature> FeatureGenerator::generate(const vector<int>& history) const {
  vector<Feature> features;
  vector<int> context;
  for (size_t i = 0; i < min(feature_context_size, history.size()); ++i) {
    int word_index = history.size() - i - 1;
    context.push_back(history[word_index]);
    features.push_back(Feature(i, context));
  }
  return features;
}

} // namespace oxlm
