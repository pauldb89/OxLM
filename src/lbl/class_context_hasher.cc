#include "lbl/class_context_hasher.h"

namespace oxlm {

ClassContextHasher::ClassContextHasher() {}

size_t ClassContextHasher::getKey(Hash context_hash) const {
  return context_hash;
}

bool ClassContextHasher::operator==(const ClassContextHasher& other) const {
  return true;
}

ClassContextHasher::~ClassContextHasher() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::ClassContextHasher)
