#include "gtest/gtest.h"

#include "lbl/factored_maxent_metadata.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(FactoredSGDTest, TestTrainMaxentSGD) {
  config->l2_maxent = 2;
  config->feature_context_size = 3;

  Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(55.079029, perplexity, EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDSparseFeatures) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;
  config->sparse_features = true;

  Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(56.51299285, perplexity, EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDCollisions) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;
  config->hash_space = 1000000;

  Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(54.080169677, perplexity, EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDExactFiltering) {
  config->l2_maxent = 0.1;
  config->feature_context_size = 3;
  config->hash_space = 1000000;
  config->filter_contexts = true;

  Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(config->test_file, dict);
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(56.5222892761, perplexity, EPS);
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
  Real objective = 0, perplexity = numeric_limits<Real>::infinity();
  model.evaluate(test_corpus, GetTime(), 0, objective, perplexity);
  EXPECT_NEAR(56.4230575561, perplexity, EPS);
}

} // namespace oxlm
