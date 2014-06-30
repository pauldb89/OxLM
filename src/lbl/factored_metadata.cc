#include "lbl/factored_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"

namespace oxlm {

FactoredMetadata::FactoredMetadata() {}

FactoredMetadata::FactoredMetadata(ModelData& config, Dict& dict)
    : Metadata(config, dict) {
  vector<int> classes;
  if (config.class_file.size()) {
    cout << "--class-file set, ignoring --classes." << endl;
    loadClassesFromFile(
        config.class_file, config.training_file, classes, dict, classBias);
  } else {
    frequencyBinning(
        config.training_file, config.classes, classes, dict, classBias);
  }

  config.vocab_size = dict.size();
  index = boost::make_shared<WordToClassIndex>(classes);
}

FactoredMetadata::FactoredMetadata(
    const ModelData& config, Dict& dict,
    const boost::shared_ptr<WordToClassIndex>& index)
    : Metadata(config, dict), index(index),
      classBias(VectorReal::Zero(index->getNumClasses())) {}

void FactoredMetadata::initialize(const boost::shared_ptr<Corpus>& corpus) {
  Metadata::initialize(corpus);
}

boost::shared_ptr<WordToClassIndex> FactoredMetadata::getIndex() const {
  return index;
}

VectorReal FactoredMetadata::getClassBias() const {
  return classBias;
}

} // namespace oxlm
