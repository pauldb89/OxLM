#include "lbl/metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/parallel_vocabulary.h"

namespace oxlm {

Metadata::Metadata() {}

Metadata::Metadata(
    const boost::shared_ptr<ModelData>& config,
    boost::shared_ptr<Vocabulary>& vocab)
    : config(config) {
  if (config->source_order == 0) {
    vocab = boost::make_shared<Vocabulary>();
  } else {
    vocab = boost::make_shared<ParallelVocabulary>();
  }
}

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
