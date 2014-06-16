#include "lbl/global_factored_maxent_weights.h"

namespace oxlm {

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata)
    : FactoredWeights(config, metadata) {}

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : FactoredWeights(config, metadata, training_corpus) {}

} // namespace oxlm
