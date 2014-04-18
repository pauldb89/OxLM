#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/feature_context_extractor.h"

namespace oxlm {

TEST(FeatureContextExtractorTest, TestLongerHistory) {
  vector<int> data = {2, 3, 4, 5, 6, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 5, 0, 1);
  FeatureContextExtractor extractor(corpus, processor, 3);

  vector<int> history = {6, 5, 4, 3, 2};
  vector<FeatureContextId> feature_context_ids =
      extractor.getFeatureContextIds(history);
  EXPECT_EQ(3, feature_context_ids.size());
  EXPECT_EQ(15, feature_context_ids[0]);
  EXPECT_EQ(16, feature_context_ids[1]);
  EXPECT_EQ(17, feature_context_ids[2]);
}

TEST(FeatureContextExtractorTest, TestShorterHistory) {
  vector<int> data = {2, 3, 4, 5, 6, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 3, 0, 1);
  FeatureContextExtractor extractor(corpus, processor, 5);

  vector<int> history = {6, 5, 4};
  vector<FeatureContextId> feature_context_ids =
      extractor.getFeatureContextIds(history);

  EXPECT_EQ(3, feature_context_ids.size());
  EXPECT_EQ(15, feature_context_ids[0]);
  EXPECT_EQ(16, feature_context_ids[1]);
  EXPECT_EQ(17, feature_context_ids[2]);
}

} // namespace oxlm
