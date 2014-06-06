#include "lbl/class_context_extractor.h"

namespace oxlm {

ClassContextExtractor::ClassContextExtractor() {}

ClassContextExtractor::ClassContextExtractor(
    const boost::shared_ptr<FeatureContextMapper>& mapper)
    : mapper(mapper) {}

vector<int> ClassContextExtractor::getFeatureContextIds(
    const vector<int>& context) const {
  return mapper->getClassContextIds(context);
}

int ClassContextExtractor::getFeatureContextId(
    const FeatureContext& feature_context) const {
  return mapper->getClassContextId(feature_context);
}

bool ClassContextExtractor::operator==(
    const ClassContextExtractor& other) const {
  return *mapper == *other.mapper;
}

ClassContextExtractor::~ClassContextExtractor() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::ClassContextExtractor)
