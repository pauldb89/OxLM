#include "lbl/class_distribution.h"

namespace oxlm {

ClassDistribution::ClassDistribution(const VectorReal& class_unigram)
    : gen(0),
      dist(class_unigram.data(), class_unigram.data() + class_unigram.size()) {}

int ClassDistribution::sample() {
  return dist(gen);
}

} // namespace oxlm
