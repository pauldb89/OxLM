#include "lbl/hash_builder.h"

namespace oxlm {

//TODO(pauldb): Add unit tests.
HashBuilder::HashBuilder(size_t seed) : seed(seed) {}

size_t HashBuilder::compose(size_t hash, int value) const {
  return hash * seed + value;
}

size_t HashBuilder::compute(const vector<int>& values) const {
  size_t ret = 1;
  for (int value: values) {
    ret = compose(ret, value);
  }
  return ret;
}

} // namespace oxlm
