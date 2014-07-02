#include "lbl/metadata.h"

namespace oxlm {

Metadata::Metadata() {}

Metadata::Metadata(const boost::shared_ptr<ModelData>& config, Dict& dict) : config(config) {}

void Metadata::initialize(const boost::shared_ptr<Corpus>& corpus) {
  unigram = VectorReal::Zero(config->vocab_size);
  for (size_t i = 0; i < corpus->size(); ++i) {
    unigram(corpus->at(i)) += 1;
  }
  unigram /= unigram.sum();
}

VectorReal Metadata::getUnigram() const {
  return unigram;
}

bool Metadata::operator==(const Metadata& other) const {
  return *config == *other.config
      && unigram == other.unigram;
}

} // namespace oxlm
