#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/class_context_hasher.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(ClassContextHasherTest, TestBasic) {
  ClassContextHasher hasher(100);
  vector<int> context = {1};
  EXPECT_EQ(54, hasher.getKey(context));
  context = {1, 2};
  EXPECT_EQ(4, hasher.getKey(context));
  context = {1, 2, 3};
  EXPECT_EQ(73, hasher.getKey(context));

  NGram expected_prediction(3, context);
  EXPECT_EQ(expected_prediction, hasher.getPrediction(3, context));
}

TEST(ClassContextHasherTest, TestSerialization) {
  boost::shared_ptr<FeatureContextHasher> hasher_ptr =
      boost::make_shared<ClassContextHasher>(100);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << hasher_ptr;

  boost::shared_ptr<FeatureContextHasher> hasher_copy_ptr;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> hasher_copy_ptr;

  boost::shared_ptr<ClassContextHasher> expected_ptr =
      dynamic_pointer_cast<ClassContextHasher>(hasher_ptr);
  boost::shared_ptr<ClassContextHasher> actual_ptr =
      dynamic_pointer_cast<ClassContextHasher>(hasher_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

} // namespace oxlm
