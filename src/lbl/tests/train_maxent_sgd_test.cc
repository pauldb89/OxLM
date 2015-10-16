#include "gtest/gtest.h"

#include "lbl/factored_maxent_metadata.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

class MaxentSGDTest : public FactoredSGDTest {
 protected:
  void SetUp() {
    FactoredSGDTest::SetUp();

    config->l2_maxent = 0.1;
    config->feature_context_size = 3;
    config->hash_space = 1000000;
  }
};

TEST_F(MaxentSGDTest, TestTrainMaxentSGDExactFiltering) {
  FactoredMaxentLM model(config);
  model.learn();
  config->test_file = "test.en";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(56.89786911, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(MaxentSGDTest, TestTrainMaxentNCE) {
  config->noise_samples = 10;

  FactoredMaxentLM model(config);
  model.learn();
  config->test_file = "test.en";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(62.33262252, perplexity(log_likelihood, test_corpus->size()), EPS);
}

} // namespace oxlm
