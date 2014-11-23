#include "gtest/gtest.h"

#include "lbl/factored_tree_weights.h"
#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(TreeSGDTest, TestTrainTreeSGD) {
  FactoredTreeLM model(config);
  model.learn();
  config->test_file = "test.en";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(68.05413818, perplexity(log_likelihood, test_corpus->size()), EPS);
}

} // namespace oxlm
