#include "lbl/ngram_query.h"

namespace oxlm {

NGramQuery::NGramQuery() {}

NGramQuery::NGramQuery(int word, const vector<int>& context) :
    word(word), classId(-1), context(context) {}

NGramQuery::NGramQuery(int word, int class_id, const vector<int>& context) :
    word(word), classId(class_id), context(context) {}

bool NGramQuery::operator==(const NGramQuery& other) const {
  return word == other.word
      && classId == other.classId
      && context == other.context;
}

} // namespace oxlm
