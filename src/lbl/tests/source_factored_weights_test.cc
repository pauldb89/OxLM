#include "gtest/gtest.h"

#include "lbl/source_factored_weights.h"

#include "utils/constants.h"

namespace oxlm {

class SourceFactoredWeightsTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 4;
    config->source_order = 2;
    config->sigmoid = true;

    vector<int> data = {2, 3, 4, 1};
    vector<int> classes = {0, 2, 4, 5};
    corpus = boost::make_shared<Corpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    metadata = boost::make_shared<FactoredMetadata>(config, dict, index);
  }

  boost::shared_ptr<ModelData> config;
  Dict dict;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FactoredMetadata> metadata;
  boost::shared_ptr<Corpus> corpus;
};

TEST_F(SourceFactoredWeightsTest, TestCheckGradient) {
  SourceFactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<SourceFactoredWeights> gradient =
      boost::make_shared<SourceFactoredWeights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, EPS));
}

/*
TEST_F(SourceFactoredWeightsTest, TestCheckGradientDiagonal) {
  config->diagonal_contexts = true;
  SourceFactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<SourceFactoredWeights> gradient =
      boost::make_shared<SourceFactoredWeights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, EPS));
}
*/

} // namespace oxlm
