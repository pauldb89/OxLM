#include "lbl/minibatch_words.h"

#include "utils/conditional_omp.h"

namespace oxlm {

void MinibatchWords::transform() {
  for (int word_id: contextWordsSet) {
    contextWords.push_back(word_id);
  }

  for (int word_id: outputWordsSet) {
    outputWords.push_back(word_id);
  }
}

void MinibatchWords::merge(const MinibatchWords& words) {
  for (int word_id: words.contextWordsSet) {
    contextWordsSet.insert(word_id);
  }

  for (int word_id: words.outputWordsSet) {
    outputWordsSet.insert(word_id);
  }
}

void MinibatchWords::addContextWord(int word_id) {
  contextWordsSet.insert(word_id);
}

void MinibatchWords::addOutputWord(int word_id) {
  outputWordsSet.insert(word_id);
}

vector<int> MinibatchWords::scatterWords(const vector<int>& words) const {
  int thread_id = omp_get_thread_num();
  int num_threads = omp_get_num_threads();

  vector<int> result;
  for (size_t i = thread_id; i < words.size(); i += num_threads) {
    result.push_back(words[i]);
  }
  return result;
}

vector<int> MinibatchWords::getContextWords() const {
  return scatterWords(contextWords);
}

vector<int> MinibatchWords::getOutputWords() const {
  return scatterWords(outputWords);
}

unordered_set<int> MinibatchWords::getContextWordsSet() const {
  return contextWordsSet;
}

unordered_set<int> MinibatchWords::getOutputWordsSet() const {
  return outputWordsSet;
}

} // namespace oxlm
