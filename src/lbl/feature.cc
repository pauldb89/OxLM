#include "feature.h"

namespace oxlm {

Feature::Feature() {}

Feature::Feature(char feature_type, const vector<int>& data) :
    feature_type(feature_type), data(data) {}

bool Feature::operator==(const Feature& feature) const {
  return feature_type == feature.feature_type && data == feature.data;
}

}
