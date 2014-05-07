#include "gtest/gtest.h"

#include "lbl/tests/test_sgd.h"
#include "lbl/train_factored_sgd.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(TestSGD, TestTrainFactoredSGD) {
  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(114.1971972, exp(-log_pp / test_corpus->size()), EPS);
}

} // namespace oxlm
