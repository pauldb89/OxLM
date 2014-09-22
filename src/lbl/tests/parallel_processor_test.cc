#include "gtest/gtest.h"

#include "lbl/parallel_processor.h"

namespace oxlm {

TEST(ParallelProcessorTest, TestBasic) {
  vector<int> source_data = {10, 11, 12, 1, 13, 14, 15, 16, 1};
  vector<int> target_data = {20, 21, 22, 1, 23, 24, 25, 1};
  vector<vector<long long>> links =
      {{1}, {0, 2}, {}, {3}, {}, {4, 6, 7}, {5}, {8}};
  boost::shared_ptr<ParallelCorpus> corpus =
      boost::make_shared<ParallelCorpus>(source_data, target_data, links);

  ParallelProcessor processor(corpus, 2, 5);
  vector<int> expected_context = {0, 0, 0, 10, 11, 12, 1};
  EXPECT_EQ(expected_context, processor.extract(0));
  expected_context = {20, 0, 0, 0, 10, 11, 12};
  EXPECT_EQ(expected_context, processor.extract(1));
  expected_context = {21, 20, 11, 12, 1, 1, 1};
  EXPECT_EQ(expected_context, processor.extract(2));
  expected_context = {22, 21, 11, 12, 1, 1, 1};
  EXPECT_EQ(expected_context, processor.extract(3));
  expected_context = {0, 0, 13, 14, 15, 16, 1};
  EXPECT_EQ(expected_context, processor.extract(4));
  expected_context = {23, 0, 13, 14, 15, 16, 1};
  EXPECT_EQ(expected_context, processor.extract(5));
  expected_context = {24, 23, 0, 13, 14, 15, 16};
  EXPECT_EQ(expected_context, processor.extract(6));
  expected_context = {25, 24, 15, 16, 1, 1, 1};
  EXPECT_EQ(expected_context, processor.extract(7));
}

TEST(ParallelProcessorTest, TestUnaligned) {
  vector<int> source_data = {2, 3, 4, 5, 6, 1};
  vector<int> target_data = {2, 3, 4, 5, 6, 1};
  vector<vector<long long>> links = {{0}, {}, {}, {}, {}, {5}};
  boost::shared_ptr<ParallelCorpus> corpus =
      boost::make_shared<ParallelCorpus>(source_data, target_data, links);

  ParallelProcessor processor(corpus, 2, 5);
  vector<int> expected_context = {0, 0, 0, 0, 2, 3, 4};
  EXPECT_EQ(expected_context, processor.extract(0));
  expected_context = {2, 0, 0, 0, 2, 3, 4};
  EXPECT_EQ(expected_context, processor.extract(1));
  expected_context = {3, 2, 0, 0, 2, 3, 4};
  EXPECT_EQ(expected_context, processor.extract(2));
  expected_context = {4, 3, 5, 6, 1, 1, 1};
  EXPECT_EQ(expected_context, processor.extract(3));
  expected_context = {5, 4, 5, 6, 1, 1, 1};
  EXPECT_EQ(expected_context, processor.extract(4));
  expected_context = {6, 5, 5, 6, 1, 1, 1};
  EXPECT_EQ(expected_context, processor.extract(5));
}

} // namespace oxlm
