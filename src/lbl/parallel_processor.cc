#include "lbl/parallel_processor.h"

namespace oxlm {

ParallelProcessor::ParallelProcessor(
    const boost::shared_ptr<Corpus>& corpus,
    int context_width, int source_context_width,
    int start_id, int end_id, int source_start_id, int source_end_id)
    : ContextProcessor(corpus, context_width, start_id, end_id),
      sourceContextSize(source_context_width), sourceStartId(source_start_id),
      sourceEndId(source_end_id) {}

/**
 * Returns the parallel conditioning context in the following format:
 * [t_{n-1}, ..., t_{n-m}, s_{a_n-sm}, s_{a_n-sm+1}, .., s_{a_n+sm}]
 * where n = current target index, a_n = t_n affinity, m = target n-gram order,
 * and sm = source order.
 */
vector<int> ParallelProcessor::extract(long long index) const {
  vector<int> context = ContextProcessor::extract(index);

  long long aligned_index = findClosestAlignedWord(index);

  boost::shared_ptr<ParallelCorpus> parallel_corpus =
      dynamic_pointer_cast<ParallelCorpus>(corpus);
  assert(parallel_corpus != nullptr);

  vector<long long> source_links = parallel_corpus->getLinks(aligned_index);
  // Round down just like in the BBN paper.
  long long affinity = source_links[(source_links.size() - 1) / 2];

  vector<int> source_context = extractSource(affinity);
  context.insert(context.end(), source_context.begin(), source_context.end());

  return context;
}

/**
 * Finds the closest target word that is aligned to at least one source word. In
 * case of a tie, the preference is given to the right.
 */
long long ParallelProcessor::findClosestAlignedWord(long long index) const {
  boost::shared_ptr<ParallelCorpus> parallel_corpus =
      dynamic_pointer_cast<ParallelCorpus>(corpus);
  assert(parallel_corpus != nullptr);

  int delta = 0;
  long long relative_index = index;
  bool overlaps_start = false, overlaps_end = false;
  // This loop is guaranteed to finish because in every sentence the
  // end-of-sentence markers are aligned.
  while (true) {
    if (index + delta < parallel_corpus->size()) {
      relative_index = index + delta;
      if (!overlaps_end && parallel_corpus->isAligned(relative_index)) {
        break;
      }

      if (parallel_corpus->at(relative_index) == endId) {
        overlaps_end = true;
      }
    }

    if (index - delta >= 0) {
      relative_index = index - delta;
      if (parallel_corpus->at(relative_index) == endId) {
        overlaps_start = true;
      }

      if (!overlaps_start && parallel_corpus->isAligned(relative_index)) {
        break;
      }
    }
    ++delta;
  }

  return relative_index;
}

vector<int> ParallelProcessor::extractSource(long long source_index) const {
  boost::shared_ptr<ParallelCorpus> parallel_corpus =
      dynamic_pointer_cast<ParallelCorpus>(corpus);
  assert(corpus != nullptr);

  vector<int> source_context;
  bool sentence_start = false;
  for (int i = 1; i <= sourceContextSize / 2; ++i) {
    long long index = source_index - i;
    sentence_start |=
        index < 0 || parallel_corpus->sourceAt(index) == sourceEndId;
    int word_id = sentence_start ?
        sourceStartId : parallel_corpus->sourceAt(index);
    source_context.push_back(word_id);
  }
  reverse(source_context.begin(), source_context.end());

  bool sentence_end = false;
  for (int i = 0; i <= sourceContextSize / 2; ++i) {
    long long index = source_index + i;
    sentence_end |=
        index >= parallel_corpus->sourceSize() ||
        parallel_corpus->sourceAt(index) == sourceEndId;
    int word_id = sentence_end ? sourceEndId : parallel_corpus->sourceAt(index);
    source_context.push_back(word_id);
  }

  return source_context;
}

} // namespace oxlm
