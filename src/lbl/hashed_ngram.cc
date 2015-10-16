#include "lbl/hashed_ngram.h"

namespace oxlm {

HashedNGram::HashedNGram(int word, int class_id, Hash context_hash)
  : word(word), classId(class_id), contextHash(context_hash) {}

} // namespace oxlm
