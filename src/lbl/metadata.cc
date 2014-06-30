#include "lbl/metadata.h"

namespace oxlm {

Metadata::Metadata() {}

Metadata::Metadata(const ModelData& config, Dict& dict) : config(config) {}

void Metadata::initialize(const boost::shared_ptr<Corpus>& corpus) {}

} // namespace oxlm
