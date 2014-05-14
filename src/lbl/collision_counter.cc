#include "lbl/collision_counter.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/class_context_keyer.h"
#include "lbl/context_processor.h"
#include "lbl/feature_context_generator.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_no_op_filter.h"
#include "lbl/word_context_extractor.h"
#include "lbl/word_context_keyer.h"

namespace oxlm {

CollisionCounter::CollisionCounter(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const boost::shared_ptr<FeatureMatcher>& matcher,
    const ModelData& config)
    : corpus(corpus), index(index), hasher(hasher), matcher(matcher),
      config(config), observedWordQueries(index->getNumClasses()) {
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, config.ngram_order - 1);
  FeatureContextGenerator generator(config.feature_context_size);
  ClassContextKeyer class_keyer(config.hash_space);

  GlobalFeatureIndexesPairPtr feature_indexes_pair;
  boost::shared_ptr<FeatureFilter> class_filter;
  if (config.filter_contexts) {
    feature_indexes_pair = matcher->getGlobalFeatures();
    class_filter = boost::make_shared<FeatureExactFilter>(
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(hasher));
  } else {
    class_filter = boost::make_shared<FeatureNoOpFilter>(
        index->getNumClasses());
  }

  vector<WordContextKeyer> word_keyers(index->getNumClasses());
  vector<boost::shared_ptr<FeatureFilter>> word_filters(index->getNumClasses());
  for (int i = 0; i < index->getNumClasses(); ++i) {
    word_keyers[i] = WordContextKeyer(i, config.hash_space);
    if (config.filter_contexts) {
      word_filters[i] = boost::make_shared<FeatureExactFilter>(
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    } else {
      word_filters[i] = boost::make_shared<FeatureNoOpFilter>(
          index->getClassSize(i));
    }
  }

  for (size_t i = 0; i < corpus->size(); ++i) {
    int class_id = index->getClass(corpus->at(i));
    vector<int> context = processor->extract(i);
    vector<FeatureContext> feature_contexts =
        generator.getFeatureContexts(context);

    for (const FeatureContext& feature_context: feature_contexts) {
      int key = class_keyer.getKey(feature_context);
      for (int index: class_filter->getIndexes(feature_context)) {
        // NGramQuery abuse: [c_n, w_{n-1}, ...]
        observedClassQueries.insert(NGramQuery(index, feature_context.data));
        observedKeys.insert((key + index) % config.hash_space);
      }

      key = word_keyers[class_id].getKey(feature_context);
      for (int index: word_filters[class_id]->getIndexes(feature_context)) {
        // NGramQuery abuse: [w_{n}, c_n, w_{n-1}, ...]; c_n is not explicit.
        observedWordQueries[class_id]
            .insert(NGramQuery(index, feature_context.data));
        observedKeys.insert((key + index) % config.hash_space);
      }
    }
  }
}

int CollisionCounter::count() const {
  int contexts = observedClassQueries.size();
  for (int i = 0; i < index->getNumClasses(); ++i) {
    contexts += observedWordQueries[i].size();
  }
  int collisions = contexts - observedKeys.size();
  cout << "Observed contexts: " << contexts << endl;
  cout << "Collisions: " << collisions << endl;
  cout << "Overall ratio: " << 100.0 * collisions / contexts << "%" << endl;

  return collisions;
}

} // namespace oxlm
