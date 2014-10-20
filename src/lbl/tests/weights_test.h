#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/weights.h"

namespace oxlm {

class WeightsTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 3;
    config->activation = SIGMOID;

    vector<int> data = {2, 3, 4, 1};
    corpus = boost::make_shared<Corpus>(data);
    metadata = boost::make_shared<Metadata>(config, vocab);
  }

  Real getLogProbabilities(
      const Weights& weights, const vector<int>& indices) const {
    Real ret = 0;
    ContextProcessor processor(corpus, config->ngram_order - 1);
    for (int i: indices) {
      vector<int> context = processor.extract(i);
      ret -= weights.getLogProb(corpus->at(i), context);
    }
    return ret;
  }

  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Vocabulary> vocab;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<Corpus> corpus;
};

} // namespace oxlm
