#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/feature.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

TEST(FeatureTest, SerializationTest) {
  vector<int> data = {1, 2, 3};
  Feature feature(2, data);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream, ar::no_header);
  output_stream << feature;

  Feature feature_copy;
  ar::binary_iarchive input_stream(stream, ar::no_header);
  input_stream >> feature_copy;

  EXPECT_EQ(feature, feature_copy);
}

}
