#pragma once

#include "lbl/factored_maxent_metadata.h"
#include "lbl/global_factored_weights.h"

namespace oxlm {

class GlobalFactoredMaxentWeights : public GlobalFactoredWeights {
 public:
  GlobalFactoredMaxentWeights(
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata);
};

} // namespace oxlm
