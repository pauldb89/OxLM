#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/weights.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

class TestWeights : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 3;
    config->sigmoid = true;

    vector<int> data = {2, 3, 4, 1};
    corpus = boost::make_shared<Corpus>(data);
    Dict dict;
    metadata = boost::make_shared<Metadata>(config, dict);
  }

  Real getPredictions(
      const Weights& weights, const vector<int>& indices) const {
    Real ret = 0;
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, config->ngram_order - 1);
    for (int index: indices) {
      vector<int> context = processor->extract(index);
      ret -= weights.predict(corpus->at(index), context);
    }
    return ret;
  }

  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<Corpus> corpus;
};

TEST_F(TestWeights, TestGradientCheck) {
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

TEST_F(TestWeights, TestGradientCheckDiagonal) {
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

TEST_F(TestWeights, TestPredict) {
  Weights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective = weights.getObjective(corpus, indices);

  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
}

TEST_F(TestWeights, TestSerialization) {
  Weights weights(config, metadata, corpus), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
