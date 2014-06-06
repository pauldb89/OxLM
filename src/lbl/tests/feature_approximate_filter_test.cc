#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_hasher.h"
#include "lbl/feature_approximate_filter.h"

namespace ar = boost::archive;

namespace oxlm {

class FeatureApproximateFilterTest : public testing::Test {
 protected:
  void SetUp() {
    hasher = boost::make_shared<ClassContextHasher>(1000);
    bloomFilter = boost::make_shared<BloomFilter<NGram>>(10, 1, 0.01);

    vector<int> context = {1, 2};
    bloomFilter->increment(NGram(1, context));
    bloomFilter->increment(NGram(3, context));
    context = {1, 5};
    bloomFilter->increment(NGram(2, context));
  }

  boost::shared_ptr<FeatureContextHasher> hasher;
  boost::shared_ptr<BloomFilter<NGram>> bloomFilter;
};

TEST_F(FeatureApproximateFilterTest, TestBasic) {
  FeatureApproximateFilter filter(5, hasher, bloomFilter);
  vector<int> context = {1, 2};
  vector<int> expected_indexes = {1, 3};
  EXPECT_EQ(expected_indexes, filter.getIndexes(context));
  context = {1, 5};
  expected_indexes = {2};
  EXPECT_EQ(expected_indexes, filter.getIndexes(context));
}

TEST_F(FeatureApproximateFilterTest, TestSerialization) {
  boost::shared_ptr<FeatureFilter> filter_ptr =
      boost::make_shared<FeatureApproximateFilter>(5, hasher, bloomFilter);

  stringstream stream(ios_base::binary | ios_base::in | ios_base::out);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << filter_ptr;

  boost::shared_ptr<FeatureFilter> filter_copy_ptr;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> filter_copy_ptr;

  boost::shared_ptr<FeatureApproximateFilter> expected_ptr =
      dynamic_pointer_cast<FeatureApproximateFilter>(filter_ptr);
  boost::shared_ptr<FeatureApproximateFilter> actual_ptr =
      dynamic_pointer_cast<FeatureApproximateFilter>(filter_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);

  vector<int> context = {1, 2};
  vector<int> expected_indexes = {1, 3};
  EXPECT_EQ(expected_indexes, filter_copy_ptr->getIndexes(context));
  context = {1, 5};
  expected_indexes = {2};
  EXPECT_EQ(expected_indexes, filter_copy_ptr->getIndexes(context));
}

} // namespace oxlm
