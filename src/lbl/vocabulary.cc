#include "lbl/vocabulary.h"

namespace oxlm {

int Vocabulary::convert(const string& word, bool frozen) {
  return dict.Convert(word, frozen);
}

string Vocabulary::convert(int word_id) {
  return dict.Convert(word_id);
}

size_t Vocabulary::size() const {
  return dict.size();
}

} // namespace oxlm
