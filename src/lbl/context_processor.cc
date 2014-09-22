#include "lbl/context_processor.h"

namespace oxlm {

ContextProcessor::ContextProcessor(
    const boost::shared_ptr<Corpus>& corpus, int context_size,
    int start_id, int end_id)
    : corpus(corpus), contextSize(context_size),
      startId(start_id), endId(end_id) {}

vector<WordId> ContextProcessor::extract(long long position) const {
  vector<WordId> context;

  // The context is constructed starting from the most recent word:
  // context = [w_{n-1}, w_{n-2}, ...]
  bool sentence_start = position == 0;
  for (int i = 1; i <= contextSize; ++i) {
    long long index = position - i;
    sentence_start |= index < 0 || corpus->at(index) == endId;
    int word_id = sentence_start ? startId : corpus->at(index);
    context.push_back(word_id);
  }

  return context;
}

} // namespace oxlm
