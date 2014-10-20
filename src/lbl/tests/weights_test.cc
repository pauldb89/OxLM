#include "lbl/tests/weights_test.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

TEST_F(WeightsTest, TestGradientCheck) {
  Weights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // In truth, using float for model parameters instead of double seriously
  // degrades the gradient computation, but has no negative effect on the
  // performance of the model and gives a 2x speed up and reduces memory by 2x.
  //
  // If you suspect there might be something off with the gradient, change
  // typedef Real to double and set a lower accepted error (e.g. 1e-5) when
  // checking the gradient.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(WeightsTest, TestGradientCheckDiagonal) {
  config->diagonal_contexts = true;

  Weights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment above if you suspect the gradient is not computed
  // correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(WeightsTest, TestGradientCheckRectifier) {
  config->activation = RECTIFIER;

  Weights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment above if you suspect the gradient is not computed
  // correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(WeightsTest, TestGetLogProb) {
  Weights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective = weights.getObjective(corpus, indices);

  EXPECT_NEAR(objective, getLogProbabilities(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getLogProbabilities(weights, indices), EPS);
}

TEST_F(WeightsTest, TestUnnormalizedScores) {
  Weights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  ContextProcessor processor(corpus, config->ngram_order - 1);
  for (int i: indices) {
    vector<int> context = processor.extract(i);

    ArrayReal scores = ArrayReal::Zero(config->vocab_size);
    for (int j = 0; j < config->vocab_size; ++j) {
      scores(j) = weights.getUnnormalizedScore(j, context);
    }

    scores = logSoftMax(scores);

    for (int j = 0; j < config->vocab_size; ++j) {
      EXPECT_NEAR(scores(j), weights.getLogProb(j, context), EPS);
    }
  }
}

TEST_F(WeightsTest, TestSerialization) {
  Weights weights(config, metadata, corpus), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
