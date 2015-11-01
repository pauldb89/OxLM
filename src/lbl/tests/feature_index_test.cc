#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/feature_index.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(FeatureIndexTest, TestBasic) {
  Hash context_hash = 0;
  FeatureIndex index;

  vector<int> expected_values;
  EXPECT_EQ(0, index.size());
  EXPECT_EQ(expected_values, index.get(context_hash));
  EXPECT_FALSE(index.contains(context_hash, 1));
  EXPECT_FALSE(index.contains(context_hash, 2));

  index.add(context_hash, 1);

  expected_values = {1};
  EXPECT_EQ(1, index.size());
  EXPECT_EQ(expected_values, index.get(context_hash));
  EXPECT_TRUE(index.contains(context_hash, 1));
  EXPECT_FALSE(index.contains(context_hash, 2));

  // Add duplicate.
  index.add(context_hash, 1);

  EXPECT_EQ(1, index.size());
  EXPECT_EQ(expected_values, index.get(context_hash));
  EXPECT_TRUE(index.contains(context_hash, 1));
  EXPECT_FALSE(index.contains(context_hash, 2));

  // Add another value.
  index.add(context_hash, 2);

  expected_values = {1, 2};
  EXPECT_EQ(1, index.size());
  EXPECT_EQ(expected_values, index.get(context_hash));
  EXPECT_TRUE(index.contains(context_hash, 1));
  EXPECT_TRUE(index.contains(context_hash, 2));

  // Check another context hash.
  context_hash = 1;
  expected_values = {};
  EXPECT_EQ(expected_values, index.get(context_hash));
  EXPECT_FALSE(index.contains(context_hash, 3));

  index.add(context_hash, 3);

  expected_values = {3};
  EXPECT_EQ(2, index.size());
  EXPECT_EQ(expected_values, index.get(context_hash));
  EXPECT_TRUE(index.contains(context_hash, 3));
}

TEST(FeatureIndexTest, TestSerialization) {
  FeatureIndex index, index_copy;
  index.add(0, 1);
  index.add(0, 2);
  index.add(1, 3);

  EXPECT_EQ(2, index.size());
  EXPECT_EQ(2, index.get(0).size());
  EXPECT_EQ(1, index.get(1).size());

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream);
  output_stream << index;

  ar::binary_iarchive input_stream(stream);
  input_stream >> index_copy;

  EXPECT_EQ(2, index_copy.size());
  vector<int> expected_values = {1, 2};
  EXPECT_EQ(expected_values, index_copy.get(0));
  EXPECT_TRUE(index_copy.contains(0, 1));
  EXPECT_TRUE(index_copy.contains(0, 2));
  EXPECT_FALSE(index_copy.contains(0, 3));

  expected_values = {3};
  EXPECT_EQ(expected_values, index_copy.get(1));
  EXPECT_FALSE(index_copy.contains(1, 1));
  EXPECT_FALSE(index_copy.contains(1, 2));
  EXPECT_TRUE(index_copy.contains(1, 3));

  expected_values = {};
  EXPECT_EQ(expected_values, index_copy.get(2));
  EXPECT_FALSE(index_copy.contains(2, 1));
  EXPECT_FALSE(index_copy.contains(2, 2));
  EXPECT_FALSE(index_copy.contains(2, 3));
}

} // namespace oxlm
