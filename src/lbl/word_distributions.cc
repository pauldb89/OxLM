#include "lbl/word_distributions.h"

namespace oxlm {

WordDistributions::WordDistributions(
    const VectorReal& unigram,
    const boost::shared_ptr<WordToClassIndex>& index)
    : index(index), gen(0) {
  for (size_t i = 0; i < index->getNumClasses(); ++i) {
    int class_start = index->getClassMarker(i);
    int class_size = index->getClassSize(i);
    dists.push_back(discrete_distribution<int>(
        unigram.data() + class_start,
        unigram.data() + class_start + class_size));
  }
}

int WordDistributions::sample(int class_id) {
  return index->getClassMarker(class_id) + dists[class_id](gen);
}

} // namespace oxlm
