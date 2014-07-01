#include "lbl/metadata.h"

namespace oxlm {

Metadata::Metadata() {}

Metadata::Metadata(const ModelData& config, Dict& dict) : config(config) {}

void Metadata::initialize(const boost::shared_ptr<Corpus>& corpus) {}

bool Metadata::operator==(const Metadata& other) const {
  return config == other.config;
}

} // namespace oxlm
