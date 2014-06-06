#include "lbl/ngram.h"

namespace oxlm {

NGram::NGram() {}

NGram::NGram(int word, const vector<int>& context) :
    word(word), classId(-1), context(context) {}

NGram::NGram(int word, int class_id, const vector<int>& context) :
    word(word), classId(class_id), context(context) {}

bool NGram::operator==(const NGram& other) const {
  return word == other.word
      && classId == other.classId
      && context == other.context;
}

} // namespace oxlm
