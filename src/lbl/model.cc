#include "lbl/model.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"
#include "lbl/utils.h"

namespace oxlm {

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Model<GlobalWeights, MinibatchWeights, Metadata>::Model(const ModelData& config)
    : config(config) {
  metadata = boost::make_shared<Metadata>(config, dict);
  weights = boost::make_shared<GlobalWeights>(metadata);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::learn() {
  boost::shared_ptr<Corpus> training_corpus =
      readCorpus(config.training_file, dict);
  if (config.test_file.size()) {
    boost::shared_ptr<Corpus> test_corpus = readCorpus(config.test_file, dict);
  }

  metadata->initialize(training_corpus);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::computeGradient() const {
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::regularize() {
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::evaluate() {
}

template class Model<GlobalWeights, MinibatchWeights, Metadata>;
template class Model<GlobalFactoredWeights, MinibatchFactoredWeights, FactoredMetadata>;
template class Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>;

} // namespace oxlm
