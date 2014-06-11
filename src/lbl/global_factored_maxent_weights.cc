#include "lbl/global_factored_maxent_weights.h"

namespace oxlm {

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
    bool model_weights)
    : FactoredWeights(config, metadata, model_weights) {}

} // namespace oxlm
