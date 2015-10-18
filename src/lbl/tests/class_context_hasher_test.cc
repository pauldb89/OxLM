#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/type_info_implementation.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/class_context_hasher.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(ClassContextHasherTest, TestBasic) {
  ClassContextHasher hasher;
  EXPECT_EQ(1, hasher.getKey(1));
  EXPECT_EQ(12, hasher.getKey(12));
  EXPECT_EQ(123, hasher.getKey(123));
}

TEST(ClassContextHasherTest, TestSerialization) {
  boost::shared_ptr<FeatureContextHasher> hasher_ptr =
      boost::make_shared<ClassContextHasher>();

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
