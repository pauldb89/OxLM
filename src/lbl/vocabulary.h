#pragma once

#include "corpus/corpus.h"
#include "lbl/archive_export.h"

using namespace std;

namespace oxlm {

class Vocabulary {
 public:
  virtual size_t size() const;

  virtual int convert(const string& word, bool frozen = false);

  virtual string convert(int word_id);

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & dict;
  }

  Dict dict;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::Vocabulary)
