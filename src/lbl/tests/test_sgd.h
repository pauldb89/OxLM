#include "gtest/gtest.h"

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"

namespace oxlm {

class TestSGD : public ::testing::Test {
 protected:
  virtual void SetUp() {
    config.training_file = "training.txt";
    config.iterations = 3;
    config.minibatch_size = 10000;
    config.instances = numeric_limits<int>::max();
    config.ngram_order = 5;
    config.l2_lbl = 2;
    config.word_representation_size = 100;
    config.threads = 1;
    config.step_size = 0.06;
    config.class_file = "classes.txt";
  }

  // TODO: This method should be refactored together with all the other places
  // where we read test corpora from files.
  Corpus loadTestCorpus(Dict dict) {
    Corpus test_corpus;
    int end_id = dict.Convert("</s>", false);
    assert(end_id >= 0);

    ifstream test_in(config.test_file);
    string line;
    while (getline(test_in, line)) {
      stringstream line_stream(line);
      string token;
      while (line_stream >> token) {
        WordId w = dict.Convert(token, true);
        if (w < 0) {
          cerr << token << " " << w << endl;
          assert(!"Unknown word found in test corpus.");
        }
        test_corpus.push_back(w);
      }
      test_corpus.push_back(end_id);
    }

    return test_corpus;
  }

  ModelData config;
};

} // namespace oxlm
