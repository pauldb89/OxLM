#include "feature_context.h"

namespace oxlm {

FeatureContext::FeatureContext() {}

FeatureContext::FeatureContext(const vector<int>& data) : data(data) {}

bool FeatureContext::operator==(const FeatureContext& feature_context) const {
  return data == feature_context.data;
}

}
