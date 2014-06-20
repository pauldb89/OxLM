#pragma once

#include <boost/shared_ptr.hpp>

#include "lbl/metadata.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FactoredMetadata : public Metadata {
 public:
  FactoredMetadata(ModelData& config, Dict& dict);

  FactoredMetadata(
      const ModelData& config, Dict& dict,
      const boost::shared_ptr<WordToClassIndex>& index);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

  boost::shared_ptr<WordToClassIndex> getIndex() const;

  VectorReal getClassBias() const;

 protected:
  VectorReal classBias;
  boost::shared_ptr<WordToClassIndex> index;
};

} // namespace oxlm
