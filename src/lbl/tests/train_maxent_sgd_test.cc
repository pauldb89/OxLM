#include "gtest/gtest.h"

#include "lbl/factored_maxent_metadata.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(FactoredSGDTest, TestTrainMaxentSGDSparseFeatures) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;

  FactoredMaxentLM model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(56.5152015, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDCollisions) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;
  config->hash_space = 1000000;

  FactoredMaxentLM model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(54.0801925, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDExactFiltering) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;
  config->hash_space = 1000000;
  config->filter_contexts = true;

  FactoredMaxentLM model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(56.5219650, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDApproximateFiltering) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;
  config->hash_space = 1000000;
  config->filter_contexts = true;
  config->filter_error_rate = 0.01;

  Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(56.4228439, perplexity(log_likelihood, test_corpus->size()), EPS);
}

} // namespace oxlm
