#pragma once

#include <random>
#include <vector>

#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class WordDistributions {
 public:
  WordDistributions(
      const VectorReal& unigram,
      const boost::shared_ptr<WordToClassIndex>& index);

  int sample(int class_id);

 private:
  boost::shared_ptr<WordToClassIndex> index;

  mt19937 gen;
  vector<discrete_distribution<int>> dists;
};

} // namespace oxlm
