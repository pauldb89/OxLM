#include "feature_context.h"

namespace oxlm {

FeatureContext::FeatureContext() {}

FeatureContext::FeatureContext(char feature_type, const vector<int>& data) :
    feature_type(feature_type), data(data) {}

bool FeatureContext::operator==(const FeatureContext& feature_context) const {
  return feature_type == feature_context.feature_type
      && data == feature_context.data;
}

}
