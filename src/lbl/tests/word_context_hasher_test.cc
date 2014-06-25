#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/word_context_hasher.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(WordContextHasherTest, TestBasic) {
  WordContextHasher hasher(13, 100);
  vector<int> context = {1};
  EXPECT_EQ(40, hasher.getKey(context));
  context = {1, 2};
  EXPECT_EQ(33, hasher.getKey(context));
  context = {1, 2, 3};
  EXPECT_EQ(18, hasher.getKey(context));

  vector<int> expected_context = {1, 2, 3};
  NGram expected_prediction(5, 13, expected_context);
  EXPECT_EQ(expected_prediction, hasher.getPrediction(5, context));
}

TEST(WordContextHasherTest, TestSerialization) {
  boost::shared_ptr<FeatureContextHasher> hasher_ptr =
      boost::make_shared<WordContextHasher>(13, 100);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << hasher_ptr;

  boost::shared_ptr<FeatureContextHasher> hasher_copy_ptr;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> hasher_copy_ptr;

  boost::shared_ptr<WordContextHasher> expected_ptr =
      dynamic_pointer_cast<WordContextHasher>(hasher_ptr);
  boost::shared_ptr<WordContextHasher> actual_ptr =
      dynamic_pointer_cast<WordContextHasher>(hasher_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

} // namespace oxlm
