#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"

namespace oxlm {

TEST(ClassContextExtractorTest, TestBasic) {
  vector<int> data = {2, 2, 2, 3, 1};
  vector<int> classes = {0, 2, 3, 4};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 2);
  boost::shared_ptr<FeatureContextHasher> hasher =
      boost::make_shared<FeatureContextHasher>(corpus, index, processor, 2);
  ClassContextExtractor extractor(hasher);

  vector<int> context = {0};
  vector<int> expected_feature_ids = {0};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  context = {0, 0};
  expected_feature_ids = {0, 1};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  context = {2};
  expected_feature_ids = {2};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  context = {2, 0};
  expected_feature_ids = {2, 3};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  context = {2, 2};
  expected_feature_ids = {2, 4};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  context = {3};
  expected_feature_ids = {5};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
  context = {3, 2};
  expected_feature_ids = {5, 6};
  EXPECT_EQ(expected_feature_ids, extractor.getFeatureContextIds(context));
}

} // namespace oxlm
