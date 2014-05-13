#include "lbl/word_context_keyer.h"

namespace oxlm {

WordContextKeyer::WordContextKeyer() {}

WordContextKeyer::WordContextKeyer(int class_id, int hash_space) :
    classId(class_id), hashSpace(hash_space) {}

int WordContextKeyer::getKey(const FeatureContext& feature_context) const {
  return hash_function(NGramQuery(classId, feature_context.data)) % hashSpace;
}

bool WordContextKeyer::operator==(const WordContextKeyer& other) const {
  return classId == other.classId && hashSpace == other.hashSpace;
}

WordContextKeyer::~WordContextKeyer() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::WordContextKeyer)
