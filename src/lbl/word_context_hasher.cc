#include "lbl/word_context_hasher.h"

namespace oxlm {

WordContextHasher::WordContextHasher() {}

WordContextHasher::WordContextHasher(
    int class_id, int hash_space_size)
    : classId(class_id), hashSpaceSize(hash_space_size) {}

int WordContextHasher::getKey(const FeatureContext& feature_context) const {
  // Note: Here we pass the class_id as the n-gram's word_id.
  // The goal is to produce a different hashcode for [context] and
  // [class_id, context].
  NGram query(classId, feature_context.data);
  return hash_function(query) % hashSpaceSize;
}

NGram WordContextHasher::getPrediction(
    int candidate, const FeatureContext& feature_context) const {
  return NGram(candidate, classId, feature_context.data);
}

bool WordContextHasher::operator==(const WordContextHasher& other) const {
  return classId == other.classId && hashSpaceSize == other.hashSpaceSize;
}

WordContextHasher::~WordContextHasher() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::WordContextHasher)
