#include "lbl/minibatch_factored_maxent_weights.h"

#include <boost/make_shared.hpp>

#include "lbl/bloom_filter.h"
#include "lbl/bloom_filter_populator.h"
#include "lbl/class_context_extractor.h"
#include "lbl/class_context_hasher.h"
#include "lbl/collision_minibatch_feature_store.h"
#include "lbl/feature_approximate_filter.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/feature_no_op_filter.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "lbl/word_context_extractor.h"
#include "lbl/word_context_hasher.h"

namespace oxlm {

MinibatchFactoredMaxentWeights::MinibatchFactoredMaxentWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata)
    : FactoredWeights(config, metadata), metadata(metadata) {
  V.resize(index->getNumClasses());

  mutexU = boost::make_shared<mutex>();
  for (int i = 0; i < V.size(); ++i) {
    mutexesV.push_back(boost::make_shared<mutex>());
  }
}

void MinibatchFactoredMaxentWeights::init(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& minibatch_indices) {
  FactoredWeights::init(corpus, minibatch_indices);

  // The number of n-gram weights updated for each minibatch is relatively low
  // compared to the number of parameters updated in the base (factored)
  // bilinear model, so it's okay to let a single thread handle this
  // initialization. Adding parallelization here is not trivial.
  #pragma omp master
  {
    int num_classes = index->getNumClasses();
    boost::shared_ptr<FeatureContextMapper> mapper = metadata->getMapper();
    boost::shared_ptr<BloomFilterPopulator> populator =
        metadata->getPopulator();
    boost::shared_ptr<FeatureMatcher> matcher = metadata->getMatcher();

    if (config->hash_space) {
      boost::shared_ptr<FeatureContextHasher> hasher =
          boost::make_shared<ClassContextHasher>(config->hash_space);
      // It's fine to use the global feature indexes here because the stores are
      // not constructed based on these indices. At filtering time, we just want
      // to know which feature indexes match which contexts.
      GlobalFeatureIndexesPairPtr feature_indexes_pair;
      boost::shared_ptr<BloomFilter<NGram>> bloom_filter;
      boost::shared_ptr<FeatureFilter> filter;
      if (config->filter_contexts) {
        if (config->filter_error_rate > 0) {
          bloom_filter = populator->get();
          filter = boost::make_shared<FeatureApproximateFilter>(
              num_classes, hasher, bloom_filter);
        } else {
          feature_indexes_pair = matcher->getGlobalFeatures();
          filter = boost::make_shared<FeatureExactFilter>(
              feature_indexes_pair->getClassIndexes(),
              boost::make_shared<ClassContextExtractor>(mapper));
        }
      } else {
        filter = boost::make_shared<FeatureNoOpFilter>(num_classes);
      }
      U = boost::make_shared<CollisionMinibatchFeatureStore>(
          num_classes, config->hash_space, config->feature_context_size,
          hasher, filter);

      for (int i = 0; i < num_classes; ++i) {
        int class_size = index->getClassSize(i);
        hasher = boost::make_shared<WordContextHasher>(
            i, config->hash_space);
        if (config->filter_contexts) {
          if (config->filter_error_rate) {
            filter = boost::make_shared<FeatureApproximateFilter>(
                class_size, hasher, bloom_filter);
          } else {
            filter = boost::make_shared<FeatureExactFilter>(
                feature_indexes_pair->getWordIndexes(i),
                boost::make_shared<WordContextExtractor>(i, mapper));
          }
        } else {
          filter = boost::make_shared<FeatureNoOpFilter>(class_size);
        }
        V[i] = boost::make_shared<CollisionMinibatchFeatureStore>(
            class_size, config->hash_space, config->feature_context_size,
            hasher, filter);
      }
    } else {
      auto feature_indexes_pair = matcher->getMinibatchFeatures(
          corpus, config->feature_context_size, minibatch_indices);
      U = boost::make_shared<SparseMinibatchFeatureStore>(
          num_classes,
          feature_indexes_pair->getClassIndexes(),
          boost::make_shared<ClassContextExtractor>(mapper));

      for (int i = 0; i < num_classes; ++i) {
        V[i] = boost::make_shared<SparseMinibatchFeatureStore>(
           index->getClassSize(i),
           feature_indexes_pair->getWordIndexes(i),
           boost::make_shared<WordContextExtractor>(i, mapper));
      }
    }
  }
}

void MinibatchFactoredMaxentWeights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient) {
  FactoredWeights::syncUpdate(words, gradient);

  {
    lock_guard<mutex> lock(*mutexU);
    U->update(gradient->U);
  }

  for (size_t i = 0; i < V.size(); ++i) {
    lock_guard<mutex> lock(*mutexesV[i]);
    V[i]->update(gradient->V[i]);
  }
}

} // namespace oxlm
