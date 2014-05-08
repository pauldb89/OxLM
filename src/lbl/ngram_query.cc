#include "lbl/ngram_query.h"

namespace oxlm {

NGramQuery::NGramQuery() {}

NGramQuery::NGramQuery(int word, const vector<int>& context) :
    word(word), context(context) {}

bool NGramQuery::operator==(const NGramQuery& other) const {
  return word == other.word && context == other.context;
}

} // namespace oxlm
