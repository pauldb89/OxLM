#include "gtest/gtest.h"

#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/weights.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(SGDTest, TestBasic) {
  Model<Weights, Weights, Metadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict, true, false);
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(72.24452209, perplexity, EPS);
}

} // namespace oxlm
