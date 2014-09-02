#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/word_distributions.h"

namespace oxlm {

TEST(WordDistributionsTest, TestBasic) {
  VectorReal class_unigrams = VectorReal::Zero(7);
  class_unigrams << 0, 1, 0.5, 0.5, 0.2, 0.5, 0.3;
  vector<int> class_markers = {0, 2, 4, 7};
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(class_markers);

  WordDistributions dists(class_unigrams, index);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(dists.sample(0) == 1);
  }

  for (int i = 0; i < 100; ++i) {
    int word_id = dists.sample(1);
    EXPECT_TRUE(2 <= word_id && word_id < 4);
  }

  for (int i = 0; i < 100; ++i) {
    int word_id = dists.sample(2);
    EXPECT_TRUE(4 <= word_id && word_id < 7);
  }
}

} // namespace oxlm
