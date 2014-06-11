#include "lbl/factored_weights.h"

namespace oxlm {

FactoredWeights::FactoredWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    bool model_weights)
    : Weights(config, metadata, model_weights) {}

} // namespace oxlm
