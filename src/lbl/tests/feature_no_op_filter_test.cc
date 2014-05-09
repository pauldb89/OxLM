#include "gtest/gtest.h"

#include <sstream>

#include "lbl/feature_no_op_filter.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace ar = boost::archive;

namespace oxlm {

TEST(FeatureNoOpFilterTest, TestBasic) {
  FeatureNoOpFilter filter(3);
  FeatureContext feature_context;
  vector<int> expected_indexes = {0, 1, 2};
  EXPECT_EQ(expected_indexes, filter.getIndexes(feature_context));
}

TEST(FeatureNoOpFilterTest, TestSerialization) {
  boost::shared_ptr<FeatureFilter> filter_ptr =
      boost::make_shared<FeatureNoOpFilter>(5);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << filter_ptr;

  boost::shared_ptr<FeatureFilter> filter_copy_ptr;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> filter_copy_ptr;

  boost::shared_ptr<FeatureNoOpFilter> expected_ptr =
      dynamic_pointer_cast<FeatureNoOpFilter>(filter_ptr);
  boost::shared_ptr<FeatureNoOpFilter> actual_ptr =
      dynamic_pointer_cast<FeatureNoOpFilter>(filter_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

} // namespace oxlm
