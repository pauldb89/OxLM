#include "lbl/tests/factored_weights_test.h"

#include "lbl/context_processor.h"
#include "lbl/factored_maxent_metadata.h"
#include "lbl/feature_matcher.h"
#include "lbl/feature_context_mapper.h"

namespace oxlm {

class GlobalFactoredMaxentWeightsTest : public FactoredWeightsTest {
 protected:
  void SetUp() {
    FactoredWeightsTest::SetUp();
    config->feature_context_size = 2;

    vector<int> data = {2, 3, 2, 4, 1};
    vector<int> classes = {0, 2, 4, 5};
    corpus = boost::make_shared<Corpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(
            corpus, config->feature_context_size);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(
            config->feature_context_size);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    mapper = boost::make_shared<FeatureContextMapper>(
        corpus, index, processor, generator, filter);
    matcher = boost::make_shared<FeatureMatcher>(
        corpus, index, processor, generator, filter, mapper);
  }

  boost::shared_ptr<FeatureContextMapper> mapper;
  boost::shared_ptr<FeatureMatcher> matcher;
  boost::shared_ptr<FactoredMaxentMetadata> metadata;
};

} // namespace oxlm
