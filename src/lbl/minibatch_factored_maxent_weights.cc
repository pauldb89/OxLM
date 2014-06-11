#include "lbl/minibatch_factored_maxent_weights.h"

namespace oxlm {

MinibatchFactoredMaxentWeights::MinibatchFactoredMaxentWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
    bool model_weights)
    : FactoredWeights(config, metadata, model_weights) {}

void MinibatchFactoredMaxentWeights::clear() {
}

} // namespace oxlm
