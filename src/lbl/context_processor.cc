#include "lbl/context_processor.h"

namespace oxlm {

ContextProcessor::ContextProcessor(
    const boost::shared_ptr<Corpus>& corpus, int context_size,
    int start_id, int end_id)
    : corpus(corpus), contextSize(context_size),
      startId(start_id), endId(end_id) {}

vector<WordId> ContextProcessor::extract(int position) const {
  vector<WordId> context;

  // The context is constructed starting from the most recent word:
  // context = [w_{n-1}, w_{n-2}, ...]
  int context_start = position - contextSize;
  bool sentence_start = position == 0;
  for (int i = contextSize - 1; i >= 0; --i) {
    int index = context_start + i;
    sentence_start |= (index < 0 || corpus->at(index) == endId);
    int word_id = sentence_start ? startId : corpus->at(index);
    context.push_back(word_id);
  }

  return context;
}

} // namespace oxlm
