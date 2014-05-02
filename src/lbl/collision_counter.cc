#include "lbl/collision_counter.h"

#include <boost/make_shared.hpp>

namespace oxlm {

CollisionCounter::CollisionCounter(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const ModelData& config)
    : corpus(corpus), index(index), config(config) {
  processor = boost::make_shared<ContextProcessor>(
      corpus, config.ngram_order - 1);
  generator = FeatureContextGenerator(config.feature_context_size);
  keyer = FeatureContextKeyer(config.hash_space, config.feature_context_size);

  for (size_t i = 0; i < corpus->size(); ++i) {
    vector<int> context = processor->extract(i);
    vector<FeatureContext> feature_contexts =
        generator.getFeatureContexts(context);
    vector<int> keys = keyer.getKeys(context);
    assert(feature_contexts.size() == keys.size());

    for (size_t i = 0; i < feature_contexts.size(); ++i) {
      observedContexts.insert(feature_contexts[i]);
      observedKeys.insert(keys[i]);
    }
  }
}

int CollisionCounter::count() const {
  cout << "Observed contexts: " << observedContexts.size() << endl;
  int collisions = observedContexts.size() - observedKeys.size();
  cout << "Collisions: " << collisions << endl;
  cout << "Ratio: " << collisions / observedContexts.size() << endl;

  return observedContexts.size() - observedKeys.size();
}

} // namespace oxlm
