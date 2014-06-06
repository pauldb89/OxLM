#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/feature_context_mapper.h"

namespace oxlm {

class FeatureContextMapperTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> data = {2, 3, 4, 5, 6, 1};
    vector<int> classes = {0, 2, 4, 7};
    corpus = boost::make_shared<Corpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    processor = boost::make_shared<ContextProcessor>(corpus, 5, 0, 1);
    generator = boost::make_shared<FeatureContextGenerator>(3);
    filter = boost::make_shared<NGramFilter>(
        corpus, index, processor, generator);
  }

  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<ContextProcessor> processor;
  boost::shared_ptr<FeatureContextGenerator> generator;
  boost::shared_ptr<NGramFilter> filter;
};

TEST_F(FeatureContextMapperTest, TestLongerHistory) {
  FeatureContextMapper mapper(corpus, index, processor, generator, filter);

  vector<int> history = {6, 5, 4, 3, 2};
  vector<int> class_context_ids = mapper.getClassContextIds(history);
  EXPECT_EQ(3, class_context_ids.size());
  EXPECT_EQ(15, class_context_ids[0]);
  EXPECT_EQ(16, class_context_ids[1]);
  EXPECT_EQ(17, class_context_ids[2]);

  vector<int> word_context_ids = mapper.getWordContextIds(0, history);
  EXPECT_EQ(3, word_context_ids.size());
  EXPECT_EQ(0, word_context_ids[0]);
  EXPECT_EQ(1, word_context_ids[1]);
  EXPECT_EQ(2, word_context_ids[2]);

  EXPECT_EQ(36, mapper.getNumContexts());
  EXPECT_EQ(18, mapper.getNumClassContexts());
  EXPECT_EQ(3, mapper.getNumWordContexts(0));
  EXPECT_EQ(6, mapper.getNumWordContexts(1));
  EXPECT_EQ(9, mapper.getNumWordContexts(2));
}

TEST_F(FeatureContextMapperTest, TestShorterHistory) {
  FeatureContextMapper mapper(corpus, index, processor, generator, filter);

  vector<int> history = {6, 5, 4};
  vector<int> class_context_ids = mapper.getClassContextIds(history);
  EXPECT_EQ(3, class_context_ids.size());
  EXPECT_EQ(15, class_context_ids[0]);
  EXPECT_EQ(16, class_context_ids[1]);
  EXPECT_EQ(17, class_context_ids[2]);

  vector<int> word_context_ids = mapper.getWordContextIds(0, history);
  EXPECT_EQ(3, word_context_ids.size());
  EXPECT_EQ(0, word_context_ids[0]);
  EXPECT_EQ(1, word_context_ids[1]);
  EXPECT_EQ(2, word_context_ids[2]);
}

} // namespace oxlm
