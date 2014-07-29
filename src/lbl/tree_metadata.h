#pragma once

#include <boost/serialization/serialization.hpp>

#include "corpus/corpus.h"
#include "lbl/class_tree.h"
#include "lbl/config.h"
#include "lbl/corpus.h"
#include "lbl/utils.h"

namespace oxlm {

class TreeMetadata {
 public:
  TreeMetadata();

  TreeMetadata(
      const boost::shared_ptr<ModelData>& config,
      boost::shared_ptr<Vocabulary>& vocab);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

  boost::shared_ptr<ClassTree> getTree() const;

  bool operator==(const TreeMetadata& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & config;
    ar & classTree;
  }

  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<ClassTree> classTree;
};

} // namespace oxlm
