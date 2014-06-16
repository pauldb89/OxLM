#pragma once

#include "lbl/factored_metadata.h"
#include "lbl/weights.h"

namespace oxlm {

class FactoredWeights : public Weights {
 public:
  FactoredWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMetadata>& metadata);

  FactoredWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);
};

} // namespace oxlm
