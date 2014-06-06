#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"

namespace ar = boost::archive;

namespace oxlm {

class ClassContextExtractorTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> data = {2, 2, 2, 3, 1};
    vector<int> classes = {0, 2, 3, 4};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 2);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(2);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    boost::shared_ptr<FeatureContextMapper> mapper =
        boost::make_shared<FeatureContextMapper>(
            corpus, index, processor, generator, filter);
    extractor = ClassContextExtractor(mapper);
  }

  ClassContextExtractor extractor;
};

TEST_F(ClassContextExtractorTest, TestBasic) {
  vector<int> context = {0};
  FeatureContext feature_context(context);
  vector<int> expected_feature_ids = {0};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(0, extractor.getFeatureContextId(feature_context));
  context = {0, 0};
  feature_context = FeatureContext(context);
  expected_feature_ids = {0, 1};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(1, extractor.getFeatureContextId(feature_context));
  context = {2};
  feature_context = FeatureContext(context);
  expected_feature_ids = {2};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(2, extractor.getFeatureContextId(feature_context));
  context = {2, 0};
  feature_context = FeatureContext(context);
  expected_feature_ids = {2, 3};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(3, extractor.getFeatureContextId(feature_context));
  context = {2, 2};
  feature_context = FeatureContext(context);
  expected_feature_ids = {2, 4};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(4, extractor.getFeatureContextId(feature_context));
  context = {3};
  feature_context = FeatureContext(context);
  expected_feature_ids = {5};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(5, extractor.getFeatureContextId(feature_context));
  context = {3, 2};
  feature_context = FeatureContext(context);
  expected_feature_ids = {5, 6};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  EXPECT_EQ(6, extractor.getFeatureContextId(feature_context));
}

TEST_F(ClassContextExtractorTest, TestSerialization) {
  boost::shared_ptr<FeatureContextExtractor> extractor_ptr =
      boost::make_shared<ClassContextExtractor>(extractor);
  boost::shared_ptr<FeatureContextExtractor> extractor_copy_ptr;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream, ar::no_header);
  output_stream << extractor_ptr;

  ar::binary_iarchive input_stream(stream, ar::no_header);
  input_stream >> extractor_copy_ptr;

  boost::shared_ptr<ClassContextExtractor> expected_ptr =
      dynamic_pointer_cast<ClassContextExtractor>(extractor_ptr);
  boost::shared_ptr<ClassContextExtractor> actual_ptr =
      dynamic_pointer_cast<ClassContextExtractor>(extractor_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

} // namespace oxlm
