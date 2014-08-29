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

  vector<int> getContextWords() const;

  vector<int> getOutputWords() const;

  unordered_set<int> getContextWordsSet() const;

  unordered_set<int> getOutputWordsSet() const;

 private:
  vector<int> scatterWords(const vector<int>& words) const;

  unordered_set<int> contextWordsSet;
  unordered_set<int> outputWordsSet;

  vector<int> contextWords;
  vector<int> outputWords;
};

} // namespace oxlm
