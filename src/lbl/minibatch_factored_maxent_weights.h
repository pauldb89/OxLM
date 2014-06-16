#pragma once

#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"

namespace oxlm {

class MinibatchFactoredMaxentWeights : public FactoredWeights {
 public:
  MinibatchFactoredMaxentWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata);

  MinibatchFactoredMaxentWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  void clear();
};

} // namespace oxlm
