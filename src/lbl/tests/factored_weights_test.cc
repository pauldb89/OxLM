#include "lbl/tests/factored_weights_test.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

TEST_F(FactoredWeightsTest, TestCheckGradient) {
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<FactoredWeights> gradient =
      boost::make_shared<FactoredWeights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);
  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(FactoredWeightsTest, TestCheckGradientDiagonal) {
  config->diagonal_contexts = true;
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<FactoredWeights> gradient =
      boost::make_shared<FactoredWeights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(FactoredWeightsTest, TestPredict) {
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  Real objective = weights.getObjective(corpus, indices);

  EXPECT_NEAR(objective, getLogProbabilities(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getLogProbabilities(weights, indices), EPS);
}

TEST_F(FactoredWeightsTest, TestUnnormalizedScores) {
  // Since we only get the sum of the word score and the class score, we can't
  // use this information to uniquely identify the original log probabilities.
  // The best I could think of is to check that the relative order of log
  // probabilities and unnormalized scores is the same.
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  EXPECT_TRUE(checkScoreRelativeOrder(weights, indices));
}

TEST_F(FactoredWeightsTest, TestSerialization) {
  FactoredWeights weights(config, metadata, corpus), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
