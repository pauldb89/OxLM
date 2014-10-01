#include "lbl/tests/weights_test.h"

#include "lbl/factored_weights.h"

namespace oxlm {

class FactoredWeightsTest : public WeightsTest {
 protected:
  void SetUp() {
    WeightsTest::SetUp();

    vector<int> classes = {0, 2, 4, 5};
    index = boost::make_shared<WordToClassIndex>(classes);
    metadata = boost::make_shared<FactoredMetadata>(config, vocab, index);
  }

  bool checkScoreRelativeOrder(
      const FactoredWeights& weights, const vector<int>& indices) const {
    ContextProcessor processor(corpus, config->ngram_order - 1);
    for (int i: indices) {
      int word_id = corpus->at(i);
      int class_id = index->getClass(word_id);
      int class_size = index->getClassSize(class_id);
      vector<int> context = processor.extract(i);

      ArrayReal scores = ArrayReal::Zero(class_size);
      ArrayReal log_probs = ArrayReal::Zero(class_size);
      for (int j = 0; j < class_size; ++j) {
        scores(j) = weights.getUnnormalizedScore(j, context);
        log_probs(j) = weights.getLogProb(j, context);
      }

      vector<int> positions(class_size);
      iota(positions.begin(), positions.end(), 0);

      sort(positions.begin(), positions.end(), [scores](int x, int y) -> bool {
        return scores[x] < scores[y];
      });

      for (int j = 1; j < class_size; ++j) {
        if (log_probs[positions[j - 1]] > log_probs[positions[j]]) {
          return false;
        }
      }
    }

    return true;
  }

  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FactoredMetadata> metadata;
};

} // namespace oxlm
