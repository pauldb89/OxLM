#pragma once

#include "lbl/factored_metadata.h"
#include "lbl/global_weights.h"

namespace oxlm {

class GlobalFactoredWeights : public GlobalWeights {
 public:
  GlobalFactoredWeights(const boost::shared_ptr<FactoredMetadata>& metadata);
};

} // namespace oxlm
