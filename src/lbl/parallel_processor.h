#pragma once

#include "lbl/context_processor.h"

namespace oxlm {

class ParallelProcessor : public ContextProcessor {
 public:
  ParallelProcessor(
      const boost::shared_ptr<Corpus>& corpus,
      int context_width, int source_context_width);

  virtual vector<int> extract(int index) const;
};

} // namespace oxlm
