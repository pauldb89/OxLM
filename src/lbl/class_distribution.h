#pragma once

#include <random>

#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class ClassDistribution {
 public:
  ClassDistribution(const VectorReal& class_unigram);

  int sample();

 private:
  mt19937 gen;
  discrete_distribution<int> dist;
};

} // namespace oxlm
