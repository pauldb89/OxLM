#pragma once

#include "lbl/corpus.h"

namespace oxlm {

class ParallelCorpus : public Corpus {
 private:
  vector<int> sourceWords;
  vector<vector<int>> links;
};

} // namespace oxlm
