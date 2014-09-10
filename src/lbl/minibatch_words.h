#pragma once

#include <unordered_set>
#include <vector>

using namespace std;

namespace oxlm {

class MinibatchWords {
 public:
  void transform();

  void merge(const MinibatchWords& words);

  void addContextWord(int word_id);

  void addOutputWord(int word_id);

  void addSourceWord(int word_id);

  vector<int> getContextWords() const;

  vector<int> getOutputWords() const;

  vector<int> getSourceWords() const;

  unordered_set<int> getContextWordsSet() const;

  unordered_set<int> getOutputWordsSet() const;

  unordered_set<int> getSourceWordsSet() const;

 private:
  vector<int> scatterWords(const vector<int>& words) const;

  void transform(const unordered_set<int>& word_set, vector<int>& words) const;

  void merge(
      unordered_set<int>& word_set,
      const unordered_set<int>& other_word_set) const;

  unordered_set<int> contextWordsSet;
  unordered_set<int> outputWordsSet;
  unordered_set<int> sourceWordsSet;

  vector<int> contextWords;
  vector<int> outputWords;
  vector<int> sourceWords;
};

} // namespace oxlm
