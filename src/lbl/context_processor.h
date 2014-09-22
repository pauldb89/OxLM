#pragma once

#include <vector>

#include "lbl/utils.h"
#include "lbl/parallel_corpus.h"

using namespace std;

namespace oxlm {

class ContextProcessor {
 public:
  ContextProcessor(
      const boost::shared_ptr<Corpus>& corpus, int context_size,
      int start_id = 0, int end_id = 1);

  virtual vector<WordId> extract(long long position) const;

 protected:
  boost::shared_ptr<Corpus> corpus;
  int contextSize, startId, endId;
};

} // namespace oxlm
