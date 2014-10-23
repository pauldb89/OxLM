#include "gtest/gtest.h"

#include "lbl/source_factored_weights.h"

#include "utils/constants.h"

namespace oxlm {

class SourceFactoredWeightsTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 6;
    config->source_vocab_size = 6;
    config->ngram_order = 4;
    config->source_order = 2;
    config->activation = SIGMOID;

    vector<int> source_data = {2, 3, 4, 5, 1};
    vector<int> target_data = {2, 3, 4, 5, 1};
    vector<vector<long long>> links = {{0}, {1}, {2, 3}, {}, {4}};
    vector<int> classes = {0, 2, 4, 6};
    corpus = boost::make_shared<ParallelCorpus>(
        source_data, target_data, links);
    index = boost::make_shared<WordToClassIndex>(classes);
    metadata = boost::make_shared<FactoredMetadata>(config, vocab, index);
  }

  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Vocabulary> vocab;
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
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

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
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(SourceFactoredWeightsTest, TestCheckGradientExtraHiddenLayers) {
  config->hidden_layers = 2;
  SourceFactoredWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<SourceFactoredWeights> gradient =
      boost::make_shared<SourceFactoredWeights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

} // namespace oxlm
