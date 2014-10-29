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
  config->test_file = "test.en";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(72.2440414, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(SGDTest, TestNCE) {
  config->noise_samples = 10;
  Model<Weights, Weights, Metadata> model(config);
  model.learn();
  config->test_file = "test.en";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(67.7363815, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(SGDTest, TestSGDExtraHiddenLayers) {
  config->hidden_layers = 2;
  config->activation = RECTIFIER;
  Model<Weights, Weights, Metadata> model(config);
  model.learn();
  config->test_file = "test.en";
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus = readTestCorpus(config, vocab);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(111.890449, perplexity(log_likelihood, test_corpus->size()), EPS);
}



} // namespace oxlm
