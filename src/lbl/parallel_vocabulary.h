#pragma once

#include "lbl/vocabulary.h"

namespace oxlm {

class ParallelVocabulary : public Vocabulary {
 public:
  int convertSource(const string& word, bool frozen=false);

  size_t sourceSize() const;

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

BOOST_CLASS_EXPORT_KEY(oxlm::ParallelVocabulary)
