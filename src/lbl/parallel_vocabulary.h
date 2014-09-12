#pragma once

#include "lbl/vocabulary.h"

namespace oxlm {

class ParallelVocabulary : public Vocabulary {
 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<Vocabulary>(*this);
    ar & sourceDict;
  }

  Dict sourceDict;
};

} // namespace oxlm
