#include "gtest/gtest.h"

#include "lbl/model.h"

namespace oxlm {

class ModelTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->training_file = "training.txt";
    config->class_file = "classes.txt";
    config->iterations = 3;
    config->minibatch_size = 10000;
    config->ngram_order = 5;
    config->l2_lbl = 2;
    config->word_representation_size = 100;
    config->threads = 1;
    config->step_size = 0.06;

    config->l2_maxent = 0.1;
    config->feature_context_size = 3;
    config->sparse_features = true;

    config->model_output_file = "model.txt";
  }

  boost::shared_ptr<ModelData> config;
};

TEST_F(ModelTest, TestSerializationCollisionStores) {
  config->hash_space = 1000000;
  config->filter_contexts = true;

  FactoredMaxentLM model(config);
  model.learn();
  model.save();

  FactoredMaxentLM model_copy;
  model_copy.load(config->model_output_file);
  EXPECT_EQ(model, model_copy);
}

TEST_F(ModelTest, TestSerializationSparseStores) {
  config->sparse_features = true;

  FactoredMaxentLM model(config);
  model.learn();
  model.save();

  FactoredMaxentLM model_copy;
  model_copy.load(config->model_output_file);
  EXPECT_EQ(model, model_copy);
}

} // namespace oxlm
