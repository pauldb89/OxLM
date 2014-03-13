#include "gtest/gtest.h"

#include <sstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/word_to_class_index.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

TEST(WordToClassIndexTest, TestBasic) {
  vector<int> class_markers = {0, 2, 3, 6};
  WordToClassIndex index(class_markers);

  EXPECT_EQ(3, index.getNumClasses());

  EXPECT_EQ(0, index.getClass(0));
  EXPECT_EQ(0, index.getClass(1));
  EXPECT_EQ(1, index.getClass(2));
  EXPECT_EQ(2, index.getClass(3));
  EXPECT_EQ(2, index.getClass(4));
  EXPECT_EQ(2, index.getClass(5));

  EXPECT_EQ(0, index.getWordIndexInClass(0));
  EXPECT_EQ(1, index.getWordIndexInClass(1));
  EXPECT_EQ(0, index.getWordIndexInClass(2));
  EXPECT_EQ(0, index.getWordIndexInClass(3));
  EXPECT_EQ(1, index.getWordIndexInClass(4));
  EXPECT_EQ(2, index.getWordIndexInClass(5));

  EXPECT_EQ(0, index.getClassMarker(0));
  EXPECT_EQ(2, index.getClassMarker(1));
  EXPECT_EQ(3, index.getClassMarker(2));
  EXPECT_EQ(6, index.getClassMarker(3));

  EXPECT_EQ(2, index.getClassSize(0));
  EXPECT_EQ(1, index.getClassSize(1));
  EXPECT_EQ(3, index.getClassSize(2));
}

TEST(WordToClassIndexTest, TestSerialization) {
  vector<int> class_markers = {0, 2, 3, 6, 10, 12};
  WordToClassIndex index(class_markers);
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream);
  output_stream << index;

  WordToClassIndex index_copy;
  ar::binary_iarchive input_stream(stream);
  input_stream >> index_copy;

  EXPECT_EQ(index, index_copy);
}

} // namespace oxlm
