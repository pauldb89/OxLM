#include "gtest/gtest.h"

#include "lbl/model.h"

namespace oxlm {

TEST(ModelTest, TestSerialization) {
  ModelData config;
  config.training_file = "training.txt";
  config.class_file = "classes.txt";
  config.iterations = 3;
  config.minibatch_size = 10000;
  config.ngram_order = 5;
  config.l2_lbl = 2;
  config.word_representation_size = 100;
  config.threads = 1;
  config.step_size = 0.06;

  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.hash_space = 1000000;
  config.filter_contexts = true;
  config.model_output_file = "model.txt";

  FactoredMaxentLM model(config);
  model.learn();
  model.save();

  FactoredMaxentLM model_copy;
  model_copy.load(config.model_output_file);
  EXPECT_EQ(model, model_copy);
}

} // namespace oxlm
