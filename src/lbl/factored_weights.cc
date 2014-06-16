#include "lbl/factored_weights.h"

namespace oxlm {

FactoredWeights::FactoredWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMetadata>& metadata)
    : Weights(config, metadata) {}

FactoredWeights::FactoredWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : Weights(config, metadata, training_corpus) {}

} // namespace oxlm
