#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/feature_context.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

TEST(FeatureContextTest, TestSerialization) {
  vector<int> data = {1, 2, 3};
  FeatureContext feature_context(data);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream, ar::no_header);
  output_stream << feature_context;

  FeatureContext feature_context_copy;
  ar::binary_iarchive input_stream(stream, ar::no_header);
  input_stream >> feature_context_copy;

  EXPECT_EQ(feature_context, feature_context_copy);
}

}
