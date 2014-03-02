#include "feature_generator.h"

namespace oxlm {

vector<Feature> FeatureGenerator::generate(const vector<int>& history) const {
  vector<Feature> features;
  vector<int> context;
  // HACK: Only consider bigrams.
  for (size_t i = 0; i < min(1, (int) history.size()); ++i) {
    int word_index = history.size() - i - 1;
    context.push_back(history[word_index]);
    features.push_back(Feature(i, context));
  }
  return features;
}

} // namespace oxlm
