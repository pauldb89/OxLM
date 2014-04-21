#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/feature_context_extractor.h"

namespace oxlm {

TEST(FeatureContextExtractorTest, TestLongerHistory) {
  vector<int> data = {2, 3, 4, 5, 6, 1};
  vector<int> classes = {0, 2, 4, 7};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 5, 0, 1);
  FeatureContextExtractor extractor(corpus, index, processor, 3);

  vector<int> history = {6, 5, 4, 3, 2};
  pair<vector<int>, vector<int>> feature_context_ids =
      extractor.getFeatureContextIds(0, history);
  EXPECT_EQ(3, feature_context_ids.first.size());
  EXPECT_EQ(15, feature_context_ids.first[0]);
  EXPECT_EQ(16, feature_context_ids.first[1]);
  EXPECT_EQ(17, feature_context_ids.first[2]);

  EXPECT_EQ(3, feature_context_ids.second.size());
  EXPECT_EQ(0, feature_context_ids.second[0]);
  EXPECT_EQ(1, feature_context_ids.second[1]);
  EXPECT_EQ(2, feature_context_ids.second[2]);
}

TEST(FeatureContextExtractorTest, TestShorterHistory) {
  vector<int> classes = {0, 2, 4, 7};
  vector<int> data = {2, 3, 4, 5, 6, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 3, 0, 1);
  FeatureContextExtractor extractor(corpus, index, processor, 5);

  vector<int> history = {6, 5, 4};
  pair<vector<int>, vector<int>> feature_context_ids =
      extractor.getFeatureContextIds(0, history);

  EXPECT_EQ(3, feature_context_ids.first.size());
  EXPECT_EQ(15, feature_context_ids.first[0]);
  EXPECT_EQ(16, feature_context_ids.first[1]);
  EXPECT_EQ(17, feature_context_ids.first[2]);

  EXPECT_EQ(3, feature_context_ids.second.size());
  EXPECT_EQ(0, feature_context_ids.second[0]);
  EXPECT_EQ(1, feature_context_ids.second[1]);
  EXPECT_EQ(2, feature_context_ids.second[2]);
}

} // namespace oxlm
