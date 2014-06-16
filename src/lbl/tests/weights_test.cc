#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/weights.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace oxlm {

class TestWeights : public testing::Test {
 protected:
  void SetUp() {
    config.word_representation_size = 3;
    config.vocab_size = 5;
    config.ngram_order = 3;
  }

  ModelData config;
  boost::shared_ptr<Metadata> metadata;
};

TEST_F(TestWeights, TestGradientCheck) {
  vector<int> data = {2, 3, 4, 1};
  vector<int> indices = {0, 1, 2, 3};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);

  Weights weights(config, metadata, corpus);
  Real objective;
  boost::shared_ptr<Weights> gradient =
      weights.getGradient(corpus, indices, objective);

  EXPECT_NEAR(6.014607708, objective, EPS);
  weights.checkGradient(corpus, indices, gradient);
}

} // namespace oxlm
