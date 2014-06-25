#include "gtest/gtest.h"

#include "lbl/tests/sgd_test.h"
#include "lbl/train_maxent_sgd.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(FactoredSGDTest, TestTrainMaxentSGD) {
  config.l2_maxent = 2;
  config.feature_context_size = 3;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(87.922121, exp(-log_pp / test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDSparseFeatures) {
  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.sparse_features = true;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(102.451838, exp(-log_pp / test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDCollisions) {
  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.hash_space = 1000000;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(81.90352573, exp(-log_pp / test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDExactFiltering) {
  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.hash_space = 1000000;
  config.filter_contexts = true;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(102.30738315, exp(-log_pp / test_corpus->size()), EPS);
}

TEST_F(FactoredSGDTest, TestTrainMaxentSGDApproximateFiltering) {
  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.hash_space = 1000000;
  config.filter_contexts = true;
  config.filter_error_rate = 0.01;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(101.558270840, exp(-log_pp / test_corpus->size()), EPS);
}

} // namespace oxlm
