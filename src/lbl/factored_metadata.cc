#include "lbl/factored_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"

namespace oxlm {

FactoredMetadata::FactoredMetadata(ModelData& config, Dict& dict)
    : Metadata(config, dict) {
  vector<int> classes;
  if (config.class_file.size()) {
    loadClassesFromFile(
        config.class_file, config.training_file, classes, dict, classBias);
  } else {
    frequencyBinning(
        config.training_file, config.classes, classes, dict, classBias);
  }

  config.vocab_size = dict.size();
  index = boost::make_shared<WordToClassIndex>(classes);
}

void FactoredMetadata::initialize(const boost::shared_ptr<Corpus>& corpus) {
  Metadata::initialize(corpus);
}

} // namespace oxlm
