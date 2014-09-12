#pragma once

#include "corpus/corpus.h"

using namespace std;

namespace oxlm {

class Vocabulary {
 public:
  size_t size() const;

  int convert(const string& word, bool frozen = false);

  string convert(int word_id);

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & dict;
  }

  Dict dict;
};

} // namespace oxlm
