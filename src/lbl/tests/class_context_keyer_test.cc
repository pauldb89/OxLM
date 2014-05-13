#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/class_context_keyer.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(ClassContextKeyerTest, TestBasic) {
  ClassContextKeyer keyer(100);
  vector<int> context = {1};
  EXPECT_EQ(54, keyer.getKey(context));
  context = {1, 2};
  EXPECT_EQ(4, keyer.getKey(context));
  context = {1, 2, 3};
  EXPECT_EQ(73, keyer.getKey(context));
}

TEST(ClassContextKeyerTest, TestSerialization) {
  boost::shared_ptr<FeatureContextKeyer> keyer_ptr =
      boost::make_shared<ClassContextKeyer>(100);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << keyer_ptr;

  boost::shared_ptr<FeatureContextKeyer> keyer_copy_ptr;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> keyer_copy_ptr;

  boost::shared_ptr<ClassContextKeyer> expected_ptr =
      dynamic_pointer_cast<ClassContextKeyer>(keyer_ptr);
  boost::shared_ptr<ClassContextKeyer> actual_ptr =
      dynamic_pointer_cast<ClassContextKeyer>(keyer_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

} // namespace oxlm
