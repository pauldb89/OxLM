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
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(61.64321517, perplexity, EPS);
}

} // namespace oxlm
