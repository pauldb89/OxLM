#include "lbl/tree_metadata.h"

#include <boost/make_shared.hpp>

namespace oxlm {

TreeMetadata::TreeMetadata() {}

TreeMetadata::TreeMetadata(
    const boost::shared_ptr<ModelData>& config,
    boost::shared_ptr<Vocabulary>& vocab)
    : Metadata(config, vocab) {
  classTree = boost::make_shared<ClassTree>(config->tree_file, vocab);
  config->vocab_size = vocab->size();
}

boost::shared_ptr<ClassTree> TreeMetadata::getTree() const {
  return classTree;
}

bool TreeMetadata::operator==(const TreeMetadata& other) const {
  return Metadata::operator==(other)
      && *classTree == *other.classTree;
}

} // namespace oxlm
