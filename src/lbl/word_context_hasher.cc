#include "lbl/word_context_hasher.h"

namespace oxlm {

WordContextHasher::WordContextHasher() {}

WordContextHasher::WordContextHasher(
    int class_id, int num_words, int hash_space_size)
    : classId(class_id), numWords(num_words), hashSpaceSize(hash_space_size) {}

int WordContextHasher::getKey(const FeatureContext& feature_context) const {
  NGram query(numWords + classId, feature_context.data);
  return hash_function(query) % hashSpaceSize;
}

NGram WordContextHasher::getPrediction(
    int candidate, const FeatureContext& feature_context) const {
  return NGram(candidate, classId, feature_context.data);
}

bool WordContextHasher::operator==(const WordContextHasher& other) const {
  return classId == other.classId
      && numWords == other.numWords
      && hashSpaceSize == other.hashSpaceSize;
}

WordContextHasher::~WordContextHasher() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::WordContextHasher)
