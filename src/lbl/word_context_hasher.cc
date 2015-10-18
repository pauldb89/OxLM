#include "lbl/word_context_hasher.h"

namespace oxlm {

WordContextHasher::WordContextHasher() {}

WordContextHasher::WordContextHasher(int class_id)
    : classId(class_id) {}

size_t WordContextHasher::getKey(Hash context_hash) const {
  // Note: Here we pass the class_id as the hashed n-gram's class_id.
  // The goal is to produce a different hashcode for [context] and
  // [class_id, context].
  return hashFunction(HashedNGram(-1, classId, context_hash));
}

bool WordContextHasher::operator==(const WordContextHasher& other) const {
  return classId == other.classId;
}

WordContextHasher::~WordContextHasher() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::WordContextHasher)
