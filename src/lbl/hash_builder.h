#pragma once

#include "lbl/utils.h"

namespace oxlm {

class HashBuilder {
 public:
  HashBuilder(size_t seed = 131687);

  size_t compose(size_t hash, int value) const;

  size_t compute(const vector<int>& values) const;

 private:
  size_t seed;
};

} // namespace oxlm
