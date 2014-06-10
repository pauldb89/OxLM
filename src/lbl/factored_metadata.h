#pragma once

#include <boost/shared_ptr.hpp>

#include "lbl/metadata.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FactoredMetadata : public Metadata {
 public:
  FactoredMetadata(const ModelData& config, Dict& dict);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

 protected:
  VectorReal classBias;
  boost::shared_ptr<WordToClassIndex> index;
};

} // namespace oxlm
