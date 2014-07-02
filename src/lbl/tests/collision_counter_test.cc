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
  boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
  config->ngram_order = 4;
  config->feature_context_size = 4;
  config->hash_space = 100;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, config->ngram_order - 1);
  boost::shared_ptr<FeatureContextGenerator> generator =
      boost::make_shared<FeatureContextGenerator>(config->feature_context_size);
  boost::shared_ptr<NGramFilter> filter =
      boost::make_shared<NGramFilter>(corpus, index, processor, generator);
  boost::shared_ptr<FeatureContextMapper> mapper =
      boost::make_shared<FeatureContextMapper>(
          corpus, index, processor, generator, filter);
  boost::shared_ptr<FeatureMatcher> matcher =
      boost::make_shared<FeatureMatcher>(
          corpus, index, processor, generator, filter, mapper);
  boost::shared_ptr<BloomFilterPopulator> populator;
  CollisionCounter counter(corpus, index, mapper, matcher, populator, config);

  EXPECT_EQ(33, counter.count());
}

} // namespace oxlm
