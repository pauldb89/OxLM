#include "gtest/gtest.h"

#include "lbl/model.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

class ConditionalSGDTest : public FactoredSGDTest {
 protected:
  void SetUp() {
    FactoredSGDTest::SetUp();

    config->source_order = 3;
    config->training_file = "training.fr-en";
    config->alignment_file = "training.gdfa";
  }
};

TEST_F(ConditionalSGDTest, TestTrainConditionalSGD) {
  SourceFactoredLM model(config);
  model.learn();
  config->test_file = "test.fr-en";
  config->test_alignment_file = "test.gdfa";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(28.1909008, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(ConditionalSGDTest, TestTrainConditionalNCE) {
  config->noise_samples = 10;

  SourceFactoredLM model(config);
  model.learn();
  config->test_file = "test.fr-en";
  config->test_alignment_file = "test.gdfa";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(49.6574172, perplexity(log_likelihood, test_corpus->size()), EPS);
}

} // namespace oxlm
