#include "lbl/minibatch_factored_maxent_weights.h"

namespace oxlm {

MinibatchFactoredMaxentWeights::MinibatchFactoredMaxentWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata)
    : FactoredWeights(config, metadata) {}

MinibatchFactoredMaxentWeights::MinibatchFactoredMaxentWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : FactoredWeights(config, metadata, training_corpus) {}

void MinibatchFactoredMaxentWeights::clear() {
}

} // namespace oxlm
