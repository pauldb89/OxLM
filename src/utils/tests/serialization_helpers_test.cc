#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "utils/constants.h"
#include "utils/serialization_helpers.h"
#include "utils/testing.h"

using namespace std;

namespace ar = boost::archive;

namespace oxlm {

TEST(SerializationHelpersTest, TestEigenMatrixFixed) {
  Eigen::Matrix<float, 2, 2> a, b;
  a << 1, 2, 3, 4;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << a;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> b;

  EXPECT_MATRIX_NEAR(a, b, EPS);
}

TEST(SerializationHelpersTest, TestEigenMatrixDynamic) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> a =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> b;
  a << 1, 2, 3, 4;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << a;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> b;

  EXPECT_MATRIX_NEAR(a, b, EPS);
}

TEST(SerializationHelpersTest, TestEigenSparseVector) {
  Eigen::SparseVector<float> a(10), b;
  a.coeffRef(3) = 5.3;
  a.coeffRef(8) = 2.5;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << a;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> b;

  EXPECT_EQ(a.size(), b.size());
  EXPECT_EQ(a.nonZeros(), b.nonZeros());
  EXPECT_NEAR(a.coeffRef(3), b.coeffRef(3), EPS);
  EXPECT_NEAR(a.coeffRef(8), b.coeffRef(8), EPS);
}

TEST(SerializationHelpersTest, TestEigenUnorderedSet) {
  unordered_set<int> a = {3, 4, 5}, b;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << a;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> b;

  EXPECT_EQ(a, b);
}

TEST(SerializationHelpersTest, TestEigenUnorderedMap) {
  unordered_map<int, int> a = {{3, 6}, {4, 5}}, b;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << a;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> b;

  EXPECT_EQ(a, b);
}

} // namespace oxlm
