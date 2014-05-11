#include "gtest/gtest.h"

#include "lbl/tests/test_sgd.h"
#include "lbl/train_maxent_sgd.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(TestSGD, TestTrainMaxentSGD) {
  config.l2_maxent = 2;
  config.feature_context_size = 3;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(87.922121, exp(-log_pp / test_corpus->size()), EPS);
}

TEST_F(TestSGD, TestTrainMaxentSGDSparseFeatures) {
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

TEST_F(TestSGD, TestTrainMaxentSGDCollisions) {
  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.hash_space = 1000000;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double log_pp = perplexity(model, test_corpus);
  EXPECT_NEAR(83.49600096, exp(-log_pp / test_corpus->size()), EPS);
}

} // namespace oxlm
