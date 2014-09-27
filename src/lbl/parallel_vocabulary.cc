#include "lbl/parallel_vocabulary.h"

namespace oxlm {

int ParallelVocabulary::convertSource(const string& word, bool frozen) {
  return sourceDict.Convert(word, frozen);
}

string ParallelVocabulary::convertSource(int word_id) {
  return sourceDict.Convert(word_id);
}

size_t ParallelVocabulary::sourceSize() const {
  return sourceDict.size();
}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::ParallelVocabulary)
