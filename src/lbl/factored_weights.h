#pragma once

#include "lbl/factored_metadata.h"
#include "lbl/weights.h"

namespace oxlm {

class FactoredWeights : public Weights {
 public:
  FactoredWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      bool model_weights = false);
};

} // namespace oxlm
