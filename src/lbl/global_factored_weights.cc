#include "lbl/global_factored_weights.h"

namespace oxlm {

GlobalFactoredWeights::GlobalFactoredWeights(
    const boost::shared_ptr<FactoredMetadata>& metadata)
    : GlobalWeights(metadata) {}

} // namespace oxlm
