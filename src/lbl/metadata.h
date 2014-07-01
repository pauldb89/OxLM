#pragma once

#include <boost/serialization/serialization.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"

namespace oxlm {

class Metadata {
 public:
  Metadata();

  Metadata(const ModelData& config, Dict& dict);

  void initialize(const boost::shared_ptr<Corpus>& corpus);

  bool operator==(const Metadata& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & config;
  }

 protected:
  ModelData config;
};

} // namespace oxlm
