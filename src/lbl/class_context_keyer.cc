#include "lbl/class_context_keyer.h"

namespace oxlm {

ClassContextKeyer::ClassContextKeyer() {}

ClassContextKeyer::ClassContextKeyer(int hash_space)
    : hashSpace(hash_space) {}

int ClassContextKeyer::getKey(const FeatureContext& feature_context) const {
  return hash_function(feature_context) % hashSpace;
}

NGramQuery ClassContextKeyer::getPrediction(
    int candidate, const FeatureContext& feature_context) const {
  return NGramQuery(candidate, feature_context.data);
}

bool ClassContextKeyer::operator==(const ClassContextKeyer& other) const {
  return hashSpace == other.hashSpace;
}

ClassContextKeyer::~ClassContextKeyer() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::ClassContextKeyer)
