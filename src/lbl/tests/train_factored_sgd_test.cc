#include "gtest/gtest.h"

#include "lbl/factored_weights.h"
#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(FactoredSGDTest, TestTrainFactoredSGD) {
  Model<FactoredWeights, FactoredWeights, FactoredMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(61.6428031, perplexity(log_likelihood, test_corpus->size()), EPS);
}

} // namespace oxlm
