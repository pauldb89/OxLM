#include "lbl/parallel_processor.h"

namespace oxlm {

ParallelProcessor::ParallelProcessor(
    const boost::shared_ptr<Corpus>& corpus,
    int context_width, int source_context_width)
    : ContextProcessor(corpus, context_width) {}

vector<int> ParallelProcessor::extract(int index) const {
  vector<int> result = ContextProcessor::extract(index);

  return result;
}

} // namespace oxlm
