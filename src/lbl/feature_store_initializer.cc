#include "lbl/feature_store_initializer.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/class_context_keyer.h"
#include "lbl/collision_global_feature_store.h"
#include "lbl/collision_minibatch_feature_store.h"
#include "lbl/collision_space.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_no_op_filter.h"
#include "lbl/sparse_global_feature_store.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "lbl/unconstrained_feature_store.h"
#include "lbl/word_context_extractor.h"
#include "lbl/word_context_keyer.h"

namespace oxlm {

FeatureStoreInitializer::FeatureStoreInitializer(
    const ModelData& config,
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const boost::shared_ptr<FeatureMatcher>& matcher)
    : config(config), index(index), hasher(hasher), matcher(matcher) {}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<GlobalFeatureStore>& U,
    vector<boost::shared_ptr<GlobalFeatureStore>>& V) const {
  if (config.hash_space) {
    // Share collision space among all stores (class + word specific).
    boost::shared_ptr<CollisionSpace> space =
        boost::make_shared<CollisionSpace>(config.hash_space);
    GlobalFeatureIndexesPairPtr feature_indexes_pair;
    boost::shared_ptr<FeatureFilter> filter;
    if (config.filter_contexts) {
      feature_indexes_pair = matcher->getGlobalFeatures();
      filter = boost::make_shared<FeatureExactFilter>(
          feature_indexes_pair->getClassIndexes(),
          boost::make_shared<ClassContextExtractor>(hasher));
    } else {
      filter = boost::make_shared<FeatureNoOpFilter>(index->getNumClasses());
    }
    U = boost::make_shared<CollisionGlobalFeatureStore>(
        index->getNumClasses(), config.hash_space,
        config.feature_context_size, space,
        boost::make_shared<ClassContextKeyer>(config.hash_space),
        filter);

    V.resize(index->getNumClasses());
    for (int i = 0; i < index->getNumClasses(); ++i) {
      if (config.filter_contexts) {
        filter = boost::make_shared<FeatureExactFilter>(
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
      } else {
        filter = boost::make_shared<FeatureNoOpFilter>(index->getClassSize(i));
      }

      V[i] = boost::make_shared<CollisionGlobalFeatureStore>(
          index->getClassSize(i), config.hash_space,
          config.feature_context_size, space,
          boost::make_shared<WordContextKeyer>(i, config.hash_space), filter);
    }
  } else if (config.sparse_features) {
    auto feature_indexes_pair = matcher->getGlobalFeatures();
    U = boost::make_shared<SparseGlobalFeatureStore>(
        index->getNumClasses(),
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(index->getNumClasses());
    for (int i = 0; i < index->getNumClasses(); ++i) {
      V[i] = boost::make_shared<SparseGlobalFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(
        index->getNumClasses(),
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(index->getNumClasses());
    for (int i = 0; i < index->getNumClasses(); ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index->getClassSize(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  }
}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<MinibatchFeatureStore>& U,
    vector<boost::shared_ptr<MinibatchFeatureStore>>& V,
    const vector<int>& minibatch_indices) const {
  if (config.hash_space) {
    // It's fine to use the global feature indexes here because the stores are
    // not constructed based on these indices. At filtering time, we just want
    // to know which feature indexes match which contexts.
    GlobalFeatureIndexesPairPtr feature_indexes_pair;
    boost::shared_ptr<FeatureFilter> filter;
    if (config.filter_contexts) {
      feature_indexes_pair = matcher->getGlobalFeatures();
      filter = boost::make_shared<FeatureExactFilter>(
          feature_indexes_pair->getClassIndexes(),
          boost::make_shared<ClassContextExtractor>(hasher));
    } else {
      filter = boost::make_shared<FeatureNoOpFilter>(index->getNumClasses());
    }
    U = boost::make_shared<CollisionMinibatchFeatureStore>(
        index->getNumClasses(),
        config.hash_space,
        config.feature_context_size,
        boost::make_shared<ClassContextKeyer>(config.hash_space),
        filter);
    V.resize(index->getNumClasses());
    for (int i = 0; i < index->getNumClasses(); ++i) {
      if (config.filter_contexts) {
        filter = boost::make_shared<FeatureExactFilter>(
            feature_indexes_pair->getWordIndexes(i),
            boost::make_shared<WordContextExtractor>(i, hasher));
      } else {
        filter = boost::make_shared<FeatureNoOpFilter>(index->getClassSize(i));
      }
      V[i] = boost::make_shared<CollisionMinibatchFeatureStore>(
          index->getClassSize(i),
          config.hash_space,
          config.feature_context_size,
          boost::make_shared<WordContextKeyer>(i, config.hash_space),
          filter);
    }
  } else if (config.sparse_features) {
    auto feature_indexes_pair = matcher->getMinibatchFeatures(minibatch_indices);
    U = boost::make_shared<SparseMinibatchFeatureStore>(
        index->getNumClasses(),
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(index->getNumClasses());
    for (int i = 0; i < index->getNumClasses(); ++i) {
      V[i] = boost::make_shared<SparseMinibatchFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(
        index->getNumClasses(),
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(index->getNumClasses());
    for (int i = 0; i < index->getNumClasses(); ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index->getClassSize(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  }
}

} // namespace oxlm
