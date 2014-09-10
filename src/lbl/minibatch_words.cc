#include "lbl/minibatch_words.h"

#include "utils/conditional_omp.h"

namespace oxlm {

void MinibatchWords::transform() {
  transform(contextWordsSet, contextWords);
  transform(outputWordsSet, outputWords);
  transform(sourceWordsSet, sourceWords);
}

void MinibatchWords::transform(
    const unordered_set<int>& word_set, vector<int>& words) const {
  for (int word_id: word_set) {
    words.push_back(word_id);
  }
}

void MinibatchWords::merge(const MinibatchWords& other) {
  merge(contextWordsSet, other.contextWordsSet);
  merge(outputWordsSet, other.outputWordsSet);
  merge(sourceWordsSet, other.sourceWordsSet);
}

void MinibatchWords::merge(
    unordered_set<int>& word_set,
    const unordered_set<int>& other_word_set) const {
  for (int word_id: other_word_set) {
    word_set.insert(word_id);
  }
}

void MinibatchWords::addContextWord(int word_id) {
  contextWordsSet.insert(word_id);
}

void MinibatchWords::addOutputWord(int word_id) {
  outputWordsSet.insert(word_id);
}

void MinibatchWords::addSourceWord(int word_id) {
  sourceWordsSet.insert(word_id);
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

vector<int> MinibatchWords::getSourceWords() const {
  return scatterWords(sourceWords);
}

unordered_set<int> MinibatchWords::getContextWordsSet() const {
  return contextWordsSet;
}

unordered_set<int> MinibatchWords::getOutputWordsSet() const {
  return outputWordsSet;
}

unordered_set<int> MinibatchWords::getSourceWordsSet() const {
  return sourceWordsSet;
}

} // namespace oxlm
