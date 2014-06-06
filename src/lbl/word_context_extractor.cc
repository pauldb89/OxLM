#include "lbl/word_context_extractor.h"

namespace oxlm {

WordContextExtractor::WordContextExtractor() {}

WordContextExtractor::WordContextExtractor(
    int class_id, const boost::shared_ptr<FeatureContextMapper>& mapper)
    : classId(class_id), mapper(mapper) {}

vector<int> WordContextExtractor::getFeatureContextIds(
    const vector<int>& context) const {
  return mapper->getWordContextIds(classId, context);
}

int WordContextExtractor::getFeatureContextId(
    const FeatureContext& feature_context) const {
  return mapper->getWordContextId(classId, feature_context);
}

bool WordContextExtractor::operator==(const WordContextExtractor& other) const {
  return classId == other.classId && *mapper == *other.mapper;
}

WordContextExtractor::~WordContextExtractor() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::WordContextExtractor)
