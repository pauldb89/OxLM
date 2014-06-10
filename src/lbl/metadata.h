#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"

namespace oxlm {

class Metadata {
 public:
  Metadata(const ModelData& config, Dict& dict);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

 protected:
  const ModelData& config;
};

} // namespace oxlm
