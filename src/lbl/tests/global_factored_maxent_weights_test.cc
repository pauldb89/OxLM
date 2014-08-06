#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/factored_maxent_metadata.h"
#include "lbl/global_factored_maxent_weights.h"
#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

class GlobalFactoredMaxentWeightsTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 3;
    config->feature_context_size = 2;
    config->sigmoid = true;

    vector<int> data = {2, 3, 2, 4, 1};
    vector<int> classes = {0, 2, 4, 5};
    corpus = boost::make_shared<Corpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(
            corpus, config->feature_context_size);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(
            config->feature_context_size);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    mapper = boost::make_shared<FeatureContextMapper>(
        corpus, index, processor, generator, filter);
    matcher = boost::make_shared<FeatureMatcher>(
        corpus, index, processor, generator, filter, mapper);
  }

  Real getPredictions(
      const GlobalFactoredMaxentWeights& weights,
      const vector<int>& indices) const {
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
  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FeatureContextMapper> mapper;
  boost::shared_ptr<BloomFilterPopulator> populator;
  boost::shared_ptr<FeatureMatcher> matcher;
  boost::shared_ptr<FactoredMaxentMetadata> metadata;
};

TEST_F(GlobalFactoredMaxentWeightsTest, TestCheckGradientUnconstrained) {
  config->sparse_features = false;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;

  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCheckGradientSparse) {
  config->sparse_features = true;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;

  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCollisionsNoFilter) {
  config->sparse_features = true;
  config->hash_space = 100;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;

  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCollisionExactFiltering) {
  config->sparse_features = true;
  config->hash_space = 100;
  config->filter_contexts = true;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;

  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestCollisionApproximateFiltering) {
  config->sparse_features = true;
  config->hash_space = 100;
  config->filter_contexts = true;
  config->filter_error_rate = 0.1;
  populator = boost::make_shared<BloomFilterPopulator>(
      corpus, index, mapper, config);
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);

  vector<int> indices = {0, 1, 2, 3, 4};
  Real objective;

  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
      weights.getGradient(corpus, indices, objective);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestPredict) {
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3};

  Real objective = weights.getObjective(corpus, indices);

  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
}

TEST_F(GlobalFactoredMaxentWeightsTest, TestSerialization) {
  config->sparse_features = false;
  metadata = boost::make_shared<FactoredMaxentMetadata>(
      config, dict, index, mapper, populator, matcher);
  GlobalFactoredMaxentWeights weights(config, metadata, corpus), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
