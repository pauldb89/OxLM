#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/global_factored_maxent_weights.h"
#include "lbl/tests/global_factored_maxent_weights_test.h"
#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

TEST_F(GlobalFactoredMaxentWeightsTest, TestCheckGradientSparse) {
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};

  Real objective;
  MinibatchWords words;
  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
       boost::make_shared<MinibatchFactoredMaxentWeights>(config, metadata);
  gradient->init(corpus, indices);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCollisionsNoFilter) {
  config->hash_space = 100;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};

  Real objective;
  MinibatchWords words;
  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
       boost::make_shared<MinibatchFactoredMaxentWeights>(config, metadata);
  gradient->init(corpus, indices);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCollisionExactFiltering) {
  config->hash_space = 100;
  config->filter_contexts = true;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
       boost::make_shared<MinibatchFactoredMaxentWeights>(config, metadata);
  gradient->init(corpus, indices);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCollisionApproximateFiltering) {
  config->hash_space = 100;
  config->filter_contexts = true;
  config->filter_error_rate = 0.1;
  populator = boost::make_shared<BloomFilterPopulator>(
      corpus, index, mapper, config);
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
       boost::make_shared<MinibatchFactoredMaxentWeights>(config, metadata);
  gradient->init(corpus, indices);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestPredict) {
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  Real objective = weights.getObjective(corpus, indices);

  EXPECT_NEAR(objective, getLogProbabilities(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getLogProbabilities(weights, indices), EPS);
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestUnnormalizedScores) {
  // Since we only get the sum of the word score and the class score, we can't
  // use this information to uniquely identify the original log probabilities.
  // The best I could think of is to check that the relative order of log
  // probabilities and unnormalized scores is the same.
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);

  GlobalFactoredMaxentWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  EXPECT_TRUE(checkScoreRelativeOrder(weights, indices));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestSerialization) {
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, vocab, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
