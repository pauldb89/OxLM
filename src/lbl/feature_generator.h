#pragma once

#include <vector>

#include "feature.h"

using namespace std;

namespace oxlm {

class FeatureGenerator {
 public:
  vector<Feature> generate(const vector<int>& history) const;
};

} // namespace oxlm
