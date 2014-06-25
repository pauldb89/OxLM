#include "lbl/word_to_class_index.h"

namespace oxlm {

WordToClassIndex::WordToClassIndex() {}

WordToClassIndex::WordToClassIndex(const vector<int>& class_markers)
    : classMarkers(class_markers), wordToClass(classMarkers.back()) {
  for (size_t i = 0; i < classMarkers.size() - 1; ++i) {
    for (int c = classMarkers[i]; c < classMarkers[i + 1]; ++c) {
      wordToClass[c] = i;
    }
  }
}

int WordToClassIndex::getNumClasses() const {
  return classMarkers.size() - 1;
}

int WordToClassIndex::getClass(int word_id) const {
  return wordToClass[word_id];
}

int WordToClassIndex::getClassMarker(int class_id) const {
  return classMarkers[class_id];
}

int WordToClassIndex::getClassSize(int class_id) const {
  return classMarkers[class_id + 1] - classMarkers[class_id];
}

int WordToClassIndex::getWordIndexInClass(int word_id) const {
  return word_id - classMarkers[wordToClass[word_id]];
}

bool WordToClassIndex::operator==(const WordToClassIndex& index) const {
  return classMarkers == index.classMarkers && wordToClass == index.wordToClass;
}

} // namespace oxlm
