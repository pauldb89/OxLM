#pragma once

#include "lbl/metadata.h"

namespace oxlm {

class GlobalWeights {
 public:
  GlobalWeights(const boost::shared_ptr<Metadata>& metadata);

 private:
  boost::shared_ptr<Metadata> metadata;
};

} // namespace oxlm
