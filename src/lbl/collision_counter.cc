#include "lbl/collision_counter.h"

#include <boost/make_shared.hpp>

#include "lbl/bloom_filter.h"
#include "lbl/class_context_extractor.h"
#include "lbl/class_context_hasher.h"
#include "lbl/context_processor.h"
#include "lbl/feature_approximate_filter.h"
#include "lbl/feature_context_generator.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_no_op_filter.h"
#include "lbl/word_context_extractor.h"
#include "lbl/word_context_hasher.h"

namespace oxlm {

CollisionCounter::CollisionCounter(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextMapper>& mapper,
    const boost::shared_ptr<FeatureMatcher>& matcher,
    const boost::shared_ptr<BloomFilterPopulator>& populator,
    const boost::shared_ptr<ModelData>& config)
    : corpus(corpus), index(index), mapper(mapper), matcher(matcher),
      populator(populator), config(config),
      observedWordQueries(index->getNumClasses()) {
  int num_classes = index->getNumClasses();
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, config->ngram_order - 1);
  FeatureContextGenerator generator(config->feature_context_size);
  boost::shared_ptr<FeatureContextHasher> class_hasher =
      boost::make_shared<ClassContextHasher>(config->hash_space);

  GlobalFeatureIndexesPairPtr feature_indexes_pair;
  boost::shared_ptr<BloomFilter<NGram>> bloom_filter;
  boost::shared_ptr<FeatureFilter> class_filter;
  if (config->filter_contexts) {
    if (config->filter_error_rate > 0) {
      bloom_filter = populator->get();
      class_filter = boost::make_shared<FeatureApproximateFilter>(
          num_classes, class_hasher, bloom_filter);
    } else {
      feature_indexes_pair = matcher->getGlobalFeatures();
      class_filter = boost::make_shared<FeatureExactFilter>(
          feature_indexes_pair->getClassIndexes(),
          boost::make_shared<ClassContextExtractor>(mapper));
    }
  } else {
    class_filter = boost::make_shared<FeatureNoOpFilter>(num_classes);
  }

  vector<boost::shared_ptr<FeatureContextHasher>> word_hashers(num_classes);
  vector<boost::shared_ptr<FeatureFilter>> word_filters(num_classes);
  for (int i = 0; i < num_classes; ++i) {
    word_hashers[i] = boost::make_shared<WordContextHasher>(
        i, config->hash_space);
    if (config->filter_contexts) {
      if (config->filter_error_rate > 0) {
        word_filters[i] = boost::make_shared<FeatureApproximateFilter>(
            index->getClassSize(i), word_hashers[i], bloom_filter);
      } else {
        word_filters[i] = boost::make_shared<FeatureExactFilter>(
            feature_indexes_pair->getWordIndexes(i),
            boost::make_shared<WordContextExtractor>(i, mapper));
      }
    } else {
      word_filters[i] = boost::make_shared<FeatureNoOpFilter>(
          index->getClassSize(i));
    }
  }

  auto start_time = GetTime();
  for (size_t i = 0; i < corpus->size(); ++i) {
    int class_id = index->getClass(corpus->at(i));
    vector<int> context = processor->extract(i);
    vector<FeatureContext> feature_contexts =
        generator.getFeatureContexts(context);

    for (const FeatureContext& feature_context: feature_contexts) {
      int key = class_hasher->getKey(feature_context);
      for (int index: class_filter->getIndexes(feature_context)) {
        observedClassQueries.insert(NGram(index, feature_context.data));
        observedKeys.insert((key + index) % config->hash_space);
      }

      key = word_hashers[class_id]->getKey(feature_context);
      for (int index: word_filters[class_id]->getIndexes(feature_context)) {
        observedWordQueries[class_id]
            .insert(NGram(index, class_id, feature_context.data));
        observedKeys.insert((key + index) % config->hash_space);
      }
    }
  }
  cout << "Counting collisions took " << GetDuration(start_time, GetTime())
       << " seconds..." << endl;
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
