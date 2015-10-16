#pragma once

#include <vector>

#include "lbl/utils.h"

namespace oxlm {

struct HashedNGram {
  HashedNGram(int word, int class_id, Hash context_hash);

  int word;
  int classId;
  Hash contextHash;
};

} // namespace oxlm

namespace std {

template<> class hash<oxlm::HashedNGram> {
 public:
   hash<oxlm::HashedNGram>(int seed = 0) : seed(seed) {}

   inline size_t operator()(const oxlm::HashedNGram& ngram) const {
     // Get the most significant 32 bits.
     int contextStart = ngram.contextHash >> 32;
     // Get the least significant 32 bits (by casting to int).
     int contextEnd = ngram.contextHash;
     vector<int> data = {ngram.word, ngram.classId, contextStart, contextEnd};
     return oxlm::MurmurHash(data, seed);
   }

  private:
   int seed;
};

} // namespace std
