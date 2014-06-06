#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/context_processor.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "lbl/word_to_class_index.h"
#include "utils/constants.h"
#include "utils/testing.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

class SparseMinibatchFeatureStoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    vector<int> data = {2, 3, 4, 5, 6};
    vector<int> classes = {0, 2, 7};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 1);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(1);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    boost::shared_ptr<FeatureContextMapper> mapper =
        boost::make_shared<FeatureContextMapper>(
            corpus, index, processor, generator, filter);
    extractor = boost::make_shared<ClassContextExtractor>(mapper);

    store = SparseMinibatchFeatureStore(5, extractor);
    SparseMinibatchFeatureStore g_store(5, extractor);

    VectorReal values(5);

    context1 = {2};
    int feature_context_id = 1;
    store.hintFeatureIndex(feature_context_id, 1);
    g_store.hintFeatureIndex(feature_context_id, 1);
    store.hintFeatureIndex(feature_context_id, 4);
    g_store.hintFeatureIndex(feature_context_id, 4);
    values << 0, 2, 0, 0, 4;
    store.update(context1, values);

    context2 = {3};
    feature_context_id = 2;
    values = SparseVectorReal(5);
    values << 1, 0, 0, 0, 3;
    store.hintFeatureIndex(feature_context_id, 0);
    store.hintFeatureIndex(feature_context_id, 1);
    store.hintFeatureIndex(feature_context_id, 4);
    store.update(context2, values);

    values = SparseVectorReal(5);
    values << 5, 3, 0, 0, 0;
    g_store.hintFeatureIndex(feature_context_id, 0);
    g_store.hintFeatureIndex(feature_context_id, 1);
    g_store.hintFeatureIndex(feature_context_id, 4);
    g_store.update(context2, values);

    context3 = {4};
    feature_context_id = 3;
    values = SparseVectorReal(5);
    values << 0, 0, 2, 1, 0;
    store.hintFeatureIndex(feature_context_id, 2);
    g_store.hintFeatureIndex(feature_context_id, 2);
    store.hintFeatureIndex(feature_context_id, 3);
    g_store.hintFeatureIndex(feature_context_id, 3);
    g_store.update(context3, values);

    gradient_store = boost::make_shared<SparseMinibatchFeatureStore>(g_store);
  }

  vector<int> context1, context2, context3;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  SparseMinibatchFeatureStore store;
  boost::shared_ptr<MinibatchFeatureStore> gradient_store;
};

TEST_F(SparseMinibatchFeatureStoreTest, TestBasic) {
  SparseMinibatchFeatureStore feature_store(5, extractor);
  EXPECT_MATRIX_NEAR(VectorReal::Zero(5), feature_store.get(context1), EPS);

  int feature_context_id = 1;
  feature_store.hintFeatureIndex(feature_context_id, 1);
  feature_store.hintFeatureIndex(feature_context_id, 3);
  feature_store.hintFeatureIndex(feature_context_id, 4);
  VectorReal values(5), expected_values(5);
  values << 10, 1, 20, 3, 4;
  expected_values << 0, 1, 0, 3, 4;
  feature_store.update(context1, values);
  EXPECT_MATRIX_NEAR(
      expected_values, feature_store.get(context1), EPS);
  feature_store.update(context1, values);
  expected_values *= 2;
  EXPECT_MATRIX_NEAR(expected_values, feature_store.get(context1), EPS);
}

TEST_F(SparseMinibatchFeatureStoreTest, TestUpdateStore) {
  store.update(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(5);
  expected_values << 0, 2, 0, 0, 4;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context1), EPS);
  expected_values << 6, 3, 0, 0, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context2), EPS);
  expected_values << 0, 0, 2, 1, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context3), EPS);
}

TEST_F(SparseMinibatchFeatureStoreTest, TestClear) {
  store.clear();
  EXPECT_EQ(0, store.size());
}

}
