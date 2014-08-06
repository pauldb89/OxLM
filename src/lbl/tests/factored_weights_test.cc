#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/factored_weights.h"
#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

class FactoredWeightsTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 3;
    config->sigmoid = true;

    vector<int> data = {2, 3, 4, 1};
    vector<int> classes = {0, 2, 4, 5};
    corpus = boost::make_shared<Corpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    metadata = boost::make_shared<FactoredMetadata>(config, dict, index);
  }

  Real getPredictions(
      const FactoredWeights& weights, const vector<int>& indices) const {
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
  Dict dict;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FactoredMetadata> metadata;
  boost::shared_ptr<Corpus> corpus;
};

TEST_F(FactoredWeightsTest, TestCheckGradient) {
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  boost::shared_ptr<FactoredWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(FactoredWeightsTest, TestCheckGradientDiagonal) {
  config->diagonal_contexts = true;
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  boost::shared_ptr<FactoredWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(FactoredWeightsTest, TestPredict) {
  FactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  Real objective = weights.getObjective(corpus, indices);

  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
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
