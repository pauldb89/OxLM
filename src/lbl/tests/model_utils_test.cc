#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"
#include "lbl/train_maxent_sgd.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

class ModelUtilsTest : public FactoredSGDTest {
 protected:
  void SetUp() {
    FactoredSGDTest::SetUp();

    vector<string> words = {"<s>", "</s>", "anna", "has", "apples", "."};
    for (const string& word: words) {
      dict.Convert(word);
    }
  }

  Dict dict;
};

TEST_F(ModelUtilsTest, TestPerplexity) {
  vector<int> classes = {0, 2, 4, 6};
  vector<int> data = {2, 3, 4, 5, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<FactoredNLM> model =
      boost::make_shared<FactoredNLM>(config, dict, index);
  EXPECT_NEAR(-8.958797, perplexity(model, corpus), EPS);
}

TEST_F(ModelUtilsTest, TestEvaluateModel) {
  Real pp = 0, best_pp = numeric_limits<Real>::infinity();
  vector<int> classes = {0, 2, 4, 6};
  vector<int> data = {2, 3, 4, 5, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<FactoredNLM> model =
      boost::make_shared<FactoredNLM>(config, dict, index);

  evaluateModel(config, model, corpus, 0, GetTime(), pp, best_pp);
  EXPECT_NEAR(6, pp, EPS);
  EXPECT_NEAR(6, best_pp, EPS);
}

TEST_F(ModelUtilsTest, TestScatterMinibatch) {
  vector<int> indices = {0, 1, 2, 3, 4, 5};

  #pragma omp parallel num_threads(2)
  {
    vector<int> result = scatterMinibatch(0, 6, indices);
    EXPECT_EQ(3, result.size());

    size_t thread_id = omp_get_thread_num();
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(2 * i + thread_id, result[i]);
    }
  }
}

TEST_F(ModelUtilsTest, TestLoadClassesFromFile) {
  vector<int> classes;
  Dict dict;
  VectorReal class_bias;
  loadClassesFromFile(
      config.class_file, config.training_file, classes, dict, class_bias);
  EXPECT_EQ(37, classes.size());
  EXPECT_EQ(0, classes[0]);
  EXPECT_EQ(2, classes[1]);
  EXPECT_EQ(8, classes[2]);
  EXPECT_EQ(17, classes[3]);
  EXPECT_EQ(1184, classes[36]);

  EXPECT_EQ(1184, dict.size());
  EXPECT_EQ(0, dict.Convert("<s>"));
  EXPECT_EQ(1, dict.Convert("</s>"));
  EXPECT_EQ(2, dict.Convert("question"));
  EXPECT_EQ(7, dict.Convert("throughout"));
  EXPECT_EQ(8, dict.Convert("limits"));
  EXPECT_EQ(1183, dict.Convert("political"));

  WordId word_id = 0;
  EXPECT_EQ("<s>", dict.Convert(word_id));
  EXPECT_EQ("</s>", dict.Convert(1));
  EXPECT_EQ("question", dict.Convert(2));
  EXPECT_EQ("throughout", dict.Convert(7));
  EXPECT_EQ("limits", dict.Convert(8));
  EXPECT_EQ("political", dict.Convert(1183));

  EXPECT_NEAR(-3.299828887, class_bias(0), EPS);
  EXPECT_NEAR(-3.617283118, class_bias(1), EPS);
  EXPECT_NEAR(-3.679626249, class_bias(2), EPS);
}

TEST_F(ModelUtilsTest, TestFrequnecyBinning) {
  vector<int> classes;
  Dict dict;
  VectorReal class_bias;
  frequencyBinning(config.training_file, 30, classes, dict, class_bias);

  EXPECT_EQ(31, classes.size());
  EXPECT_EQ(0, classes[0]);
  EXPECT_EQ(2, classes[1]);
  EXPECT_EQ(3, classes[2]);
  EXPECT_EQ(4, classes[3]);
  EXPECT_EQ(5, classes[4]);
  EXPECT_EQ(1184, classes[30]);

  EXPECT_EQ(1184, dict.size());
  EXPECT_EQ(0, dict.Convert("<s>"));
  EXPECT_EQ(1, dict.Convert("</s>"));
  EXPECT_EQ(2, dict.Convert("<unk>"));
  EXPECT_EQ(3, dict.Convert("the"));
  EXPECT_EQ(4, dict.Convert(","));

  WordId word_id = 0;
  EXPECT_EQ("<s>", dict.Convert(word_id));
  EXPECT_EQ("</s>", dict.Convert(1));
  EXPECT_EQ("<unk>", dict.Convert(2));
  EXPECT_EQ("the", dict.Convert(3));
  EXPECT_EQ(",", dict.Convert(4));

  EXPECT_NEAR(-3.299828887, class_bias(0), EPS);
  EXPECT_NEAR(-2.021676685, class_bias(1), EPS);
  EXPECT_NEAR(-2.841139018, class_bias(2), EPS);
  EXPECT_NEAR(-3.035927343, class_bias(3), EPS);
}

TEST_F(ModelUtilsTest, TestSerializeModel) {
  config.l2_maxent = 0.1;
  config.feature_context_size = 3;
  config.sparse_features = true;

  boost::shared_ptr<FactoredNLM> model = learn(config);
  config.test_file = "test.txt";
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, model->label_set());
  double expected_perplexity = perplexity(model, test_corpus);

  saveModel("model.bin", model);
  boost::shared_ptr<FactoredNLM> model_copy =
      loadModel("model.bin", test_corpus);

  EXPECT_NEAR(expected_perplexity, perplexity(model_copy, test_corpus), EPS);
}

} // namespace oxlm
