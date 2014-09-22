#pragma once

#include "lbl/context_processor.h"

namespace oxlm {

class ParallelProcessor : public ContextProcessor {
 public:
  ParallelProcessor(
      const boost::shared_ptr<Corpus>& corpus,
      int context_width, int source_context_width,
      int start_id = 0, int end_id = 1,
      int source_start_id = 0, int source_end_id = 1);

  virtual vector<int> extract(long long index) const;

 private:
  long long findClosestAlignedWord(long long index) const;

  vector<int> extractSource(long long source_index) const;

  int sourceContextSize, sourceStartId, sourceEndId;
};

} // namespace oxlm
