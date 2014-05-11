#include "lbl/class_hash_space_decider.h"

#include "lbl/utils.h"

namespace oxlm {

ClassHashSpaceDecider::ClassHashSpaceDecider(
    const boost::shared_ptr<WordToClassIndex>& index, int hash_space)
    : classHashSpaces(index->getNumClasses()) {
  double sum = 0;
  for (int i = 0; i < index->getNumClasses(); ++i) {
    sum += pow(index->getClassSize(i), 0.333);
  }

  for (int i = 0; i < index->getNumClasses(); ++i) {
    classHashSpaces[i] = pow(index->getClassSize(i), 0.333) / sum * hash_space;
  }
}

int ClassHashSpaceDecider::getHashSpace(int class_id) const {
  return classHashSpaces[class_id];
}

} // namespace oxlm
