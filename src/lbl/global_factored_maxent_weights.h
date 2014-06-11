#pragma once

#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"

namespace oxlm {

class GlobalFactoredMaxentWeights : public FactoredWeights {
 public:
  GlobalFactoredMaxentWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
      bool model_weights = false);
};

} // namespace oxlm
