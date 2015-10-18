#include "lbl/minibatch_factored_maxent_weights.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_hasher.h"
#include "lbl/feature_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/minibatch_feature_store.h"
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
  int num_classes = index->getNumClasses();
  boost::shared_ptr<FeatureMatcher> matcher = metadata->getMatcher();

  boost::shared_ptr<FeatureContextHasher> hasher =
      boost::make_shared<ClassContextHasher>();
  // It's fine to use the global feature indexes here because the stores are
  // not constructed based on these indices. At filtering time, we just want
  // to know which feature indexes match which contexts.
  FeatureIndexesPairPtr feature_indexes_pair;
  boost::shared_ptr<FeatureFilter> filter;
  feature_indexes_pair = matcher->getFeatureIndexes();
  filter = boost::make_shared<FeatureFilter>(
      feature_indexes_pair->getClassIndexes());
  U = boost::make_shared<MinibatchFeatureStore>(
      num_classes, config->hash_space, config->feature_context_size,
      hasher, filter);

  for (int i = 0; i < num_classes; ++i) {
    int class_size = index->getClassSize(i);
    hasher = boost::make_shared<WordContextHasher>(i);
    filter = boost::make_shared<FeatureFilter>(
        feature_indexes_pair->getWordIndexes(i));
    V[i] = boost::make_shared<MinibatchFeatureStore>(
        class_size, config->hash_space, config->feature_context_size,
        hasher, filter);
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
