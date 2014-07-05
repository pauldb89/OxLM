#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"

namespace oxlm {

class SGDTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    config = boost::make_shared<ModelData>();
    config->training_file = "training.txt";
    config->iterations = 3;
    config->minibatch_size = 10000;
    config->ngram_order = 5;
    config->l2_lbl = 2;
    config->word_representation_size = 100;
    config->threads = 1;
    config->step_size = 0.06;
    config->sigmoid = true;
  }

  boost::shared_ptr<ModelData> config;
};

class FactoredSGDTest : public SGDTest {
 protected:
  virtual void SetUp() {
    SGDTest::SetUp();
    config->class_file = "classes.txt";
  }
};

} // namespace oxlm
