#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/collision_counter.h"

namespace oxlm {

TEST(CollisionCounterTest, TestBasic) {
  vector<int> data = {3, 4, 2, 3, 1, 5, 3};
  vector<int> classes = {0, 2, 3, 6};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  ModelData config;
  config.ngram_order = 4;
  config.feature_context_size = 4;
  config.hash_space = 100;
  CollisionCounter counter(corpus, index, config);

  EXPECT_EQ(25, counter.count());
}

} // namespace oxlm
