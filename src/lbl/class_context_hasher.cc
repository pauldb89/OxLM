#include "lbl/class_context_hasher.h"

namespace oxlm {

ClassContextHasher::ClassContextHasher() {}

ClassContextHasher::ClassContextHasher(int hash_space)
    : hashSpace(hash_space) {}

int ClassContextHasher::getKey(const FeatureContext& feature_context) const {
  return hash_function(feature_context) % hashSpace;
}

NGram ClassContextHasher::getPrediction(
    int candidate, const FeatureContext& feature_context) const {
  return NGram(candidate, feature_context.data);
}

bool ClassContextHasher::operator==(const ClassContextHasher& other) const {
  return hashSpace == other.hashSpace;
}

ClassContextHasher::~ClassContextHasher() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::ClassContextHasher)
