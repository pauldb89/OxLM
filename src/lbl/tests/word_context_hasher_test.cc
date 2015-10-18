#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/type_info_implementation.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/word_context_hasher.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(WordContextHasherTest, TestBasic) {
  WordContextHasher hasher(13);
  EXPECT_EQ(16608802623032811381ULL, hasher.getKey(1));
  EXPECT_EQ(3381112863320154404ULL, hasher.getKey(12));
  EXPECT_EQ(2692707420317967352ULL, hasher.getKey(123));
}

TEST(WordContextHasherTest, TestSerialization) {
  boost::shared_ptr<FeatureContextHasher> hasher_ptr =
      boost::make_shared<WordContextHasher>(13);

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
