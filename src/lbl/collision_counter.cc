#include "lbl/collision_counter.h"

#include <boost/make_shared.hpp>

namespace oxlm {

CollisionCounter::CollisionCounter(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const ModelData& config)
    : corpus(corpus), index(index), config(config),
      observedWordContexts(index->getNumClasses()),
      observedWordKeys(index->getNumClasses()) {
  processor = boost::make_shared<ContextProcessor>(
      corpus, config.ngram_order - 1);
  generator = FeatureContextGenerator(config.feature_context_size);
  class_keyer = FeatureContextKeyer(
      config.hash_space, config.feature_context_size);
  int class_hash_space = config.hash_space / index->getNumClasses();
  word_keyer = FeatureContextKeyer(
      class_hash_space, config.feature_context_size);

  for (size_t i = 0; i < corpus->size(); ++i) {
    int class_id = index->getClass(corpus->at(i));
    vector<int> context = processor->extract(i);
    vector<FeatureContext> feature_contexts =
        generator.getFeatureContexts(context);

    vector<int> class_keys = class_keyer.getKeys(context);
    assert(feature_contexts.size() == class_keys.size());
    for (size_t i = 0; i < feature_contexts.size(); ++i) {
      observedClassContexts.insert(feature_contexts[i]);
      for (int j = 0; j < index->getNumClasses(); ++j) {
        observedClassKeys.insert((class_keys[i] + j) % config.hash_space);
      }
    }

    vector<int> word_keys = word_keyer.getKeys(context);
    assert(feature_contexts.size() == word_keys.size());
    for (size_t i = 0; i < feature_contexts.size(); ++i) {
      observedWordContexts[class_id].insert(feature_contexts[i]);
      for (int j = 0; j < index->getClassSize(class_id); ++j) {
        observedWordKeys[class_id].insert(
            (word_keys[i] + j) % class_hash_space);
      }
    }
  }
}

int CollisionCounter::count() const {
  int class_contexts = observedClassContexts.size() * index->getNumClasses();
  int class_collisions = class_contexts - observedClassKeys.size();
  cout << "Observed class contexts: " << class_contexts << endl;
  cout << "Class collisions: " << class_collisions << endl;
  cout << "Ratio (for classes): "
       << Real(class_collisions) / class_contexts << endl;

  int word_contexts_total = 0, word_collisions = 0;
  for (int i = 0; i < index->getNumClasses(); ++i) {
    int word_contexts = observedWordContexts[i].size() * index->getClassSize(i);
    word_contexts_total += word_contexts;
    word_collisions += word_contexts - observedWordKeys[i].size();
  }
  cout << "Observed word contexts: " << word_contexts_total << endl;
  cout << "Word collisions: " << word_collisions << endl;
  cout << "Ratio (for words): "
       << Real(word_collisions) / word_contexts_total << endl;

  int contexts = class_contexts + word_contexts_total;
  int collisions = class_collisions + word_collisions;
  cout << "Observed contexts: " << contexts << endl;
  cout << "Collisions: " << collisions << endl;
  cout << "Overall ratio: " << Real(collisions) / contexts << endl;

  return collisions;
}

} // namespace oxlm
