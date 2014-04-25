#include "lbl/feature_store_initializer.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/sparse_global_feature_store.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "lbl/unconstrained_feature_store.h"
#include "lbl/word_context_extractor.h"

namespace oxlm {

FeatureStoreInitializer::FeatureStoreInitializer(
    const ModelData& config,
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index)
    : config(config), index(index) {
  int context_width = config.ngram_order - 1;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, context_width);
  hasher = boost::make_shared<FeatureContextHasher>(
      corpus, index, processor, config.feature_context_size);
  matcher = boost::make_shared<FeatureMatcher>(
      corpus, index, processor, hasher);
}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<GlobalFeatureStore>& U,
    vector<boost::shared_ptr<GlobalFeatureStore>>& V) const {
  if (config.sparse_features) {
    auto feature_indexes_pair = matcher->getGlobalFeatures();
    U = boost::make_shared<SparseGlobalFeatureStore>(
        config.classes,
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<SparseGlobalFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(
        config.classes,
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
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
  if (config.sparse_features) {
    auto feature_indexes_pair = matcher->getMinibatchFeatures(minibatch_indices);
    U = boost::make_shared<SparseMinibatchFeatureStore>(
        config.classes,
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<SparseMinibatchFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(
        config.classes,
        boost::make_shared<ClassContextExtractor>(hasher));
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index->getClassSize(i),
          boost::make_shared<WordContextExtractor>(i, hasher));
    }
  }
}

} // namespace oxlm
