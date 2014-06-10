#include "lbl/global_factored_maxent_weights.h"

namespace oxlm {

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata)
    : GlobalFactoredWeights(metadata) {}

} // namespace oxlm
