#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/class_hash_space_decider.h"

namespace oxlm {

TEST(ClassHashSpaceDeciderTest, TestBasic) {
  vector<int> classes = {0, 2, 4, 5, 7, 10};
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  ClassHashSpaceDecider decider(index, 1000);
  EXPECT_EQ(200, decider.getHashSpace(0));
  EXPECT_EQ(200, decider.getHashSpace(1));
  EXPECT_EQ(100, decider.getHashSpace(2));
  EXPECT_EQ(200, decider.getHashSpace(3));
  EXPECT_EQ(300, decider.getHashSpace(4));
}

} // namespace oxlm
