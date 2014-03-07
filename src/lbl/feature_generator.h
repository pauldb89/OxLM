#pragma once

#include <vector>

#include "feature.h"

using namespace std;

namespace oxlm {

class FeatureGenerator {
 public:
  FeatureGenerator(size_t feature_context_size);

  vector<Feature> generate(const vector<int>& history) const;

 private:
  size_t feature_context_size;
};

} // namespace oxlm
