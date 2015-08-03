#include "lbl/global_factored_maxent_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/class_context_hasher.h"
#include "lbl/collision_global_feature_store.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/sparse_global_feature_store.h"
#include "lbl/word_context_extractor.h"
#include "lbl/word_context_hasher.h"

namespace oxlm {

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights() {}

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata)
    : FactoredWeights(config, metadata), metadata(metadata) {
  initialize();
}

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : FactoredWeights(config, metadata, training_corpus), metadata(metadata) {
  initialize();
}

void GlobalFactoredMaxentWeights::initialize() {
  int num_classes = index->getNumClasses();
  V.resize(num_classes);
  boost::shared_ptr<FeatureContextMapper> mapper = metadata->getMapper();
  boost::shared_ptr<FeatureMatcher> matcher = metadata->getMatcher();

  if (config->hash_space) {
    boost::shared_ptr<GlobalCollisionSpace> space =
        boost::make_shared<GlobalCollisionSpace>(config->hash_space);
    boost::shared_ptr<FeatureContextHasher> hasher =
        boost::make_shared<ClassContextHasher>(config->hash_space);
    GlobalFeatureIndexesPairPtr feature_indexes_pair;
    boost::shared_ptr<FeatureFilter> filter;
    feature_indexes_pair = matcher->getGlobalFeatures();
    filter = boost::make_shared<FeatureExactFilter>(
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(mapper));

    U = boost::make_shared<CollisionGlobalFeatureStore>(
        num_classes, config->hash_space, config->feature_context_size,
        space, hasher, filter);

    for (int i = 0; i < num_classes; ++i) {
      int class_size = index->getClassSize(i);
      hasher = boost::make_shared<WordContextHasher>(i, config->hash_space);
      filter = boost::make_shared<FeatureExactFilter>(
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, mapper));
      V[i] = boost::make_shared<CollisionGlobalFeatureStore>(
          class_size, config->hash_space, config->feature_context_size,
          space, hasher, filter);
    }
  } else {
    auto feature_indexes_pair = matcher->getGlobalFeatures();
    U = boost::make_shared<SparseGlobalFeatureStore>(
        num_classes,
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(mapper));

    for (int i = 0; i < num_classes; ++i) {
      V[i] = boost::make_shared<SparseGlobalFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, mapper));
    }
  }
}

size_t GlobalFactoredMaxentWeights::numParameters() const {
  // TODO: Count parameters in feature stores.
  return FactoredWeights::numParameters();
}

void GlobalFactoredMaxentWeights::getProbabilities(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& forward_weights,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  class_probs = S.transpose() * forward_weights.back() + T * MatrixReal::Ones(1, indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);

    VectorReal prediction_vector = forward_weights.back().col(i);
    VectorReal class_scores = class_probs.col(i) + U->get(contexts[i]);
    class_probs.col(i) = softMax(class_scores);

    VectorReal word_scores = classR(class_id).transpose() * prediction_vector +
                             classB(class_id) + V[class_id]->get(contexts[i]);
    word_probs.push_back(softMax(word_scores));
  }
}

void GlobalFactoredMaxentWeights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  vector<MatrixReal> forward_weights;
  MatrixReal class_probs;
  vector<VectorReal> word_probs;
  log_likelihood += getObjective(
      corpus, indices, contexts, context_vectors, forward_weights,
      class_probs, word_probs);

  setContextWords(contexts, words);

  getFullGradient(
      corpus, indices, contexts, context_vectors, forward_weights,
      class_probs, word_probs, gradient, words);
}

void GlobalFactoredMaxentWeights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const vector<MatrixReal>& forward_weights,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
    MinibatchWords& words) const {
  FactoredWeights::getFullGradient(
      corpus, indices, contexts, context_vectors, forward_weights,
      class_probs, word_probs, gradient, words);

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);

    gradient->U->update(contexts[i], class_probs.col(i));
    gradient->V[class_id]->update(contexts[i], word_probs[i]);
  }
}

bool GlobalFactoredMaxentWeights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
    Real eps) {
  if (!FactoredWeights::checkGradient(corpus, indices, gradient, eps)) {
    return false;
  }

  if (config->hash_space == 0) {
    // If no hashing is used, check gradients individually.
    if (!checkGradientStore(corpus, indices, U, gradient->U, eps)) {
      return false;
    }

    for (size_t i = 0; i < V.size(); ++i) {
      if (!checkGradientStore(corpus, indices, V[i], gradient->V[i], eps)) {
        return false;
      }
    }
  } else {
    // Class and word features are hashed in the same global collision space, so
    // we can only verify if the sum of gradients is correct.
    boost::shared_ptr<MinibatchFeatureStore> gradient_sum = gradient->U;
    for (size_t i = 0; i < V.size(); ++i) {
      gradient_sum->update(gradient->V[i]);
    }

    if (!checkGradientStore(corpus, indices, U, gradient_sum, eps)) {
      return false;
    }
  }

  return true;
}

bool GlobalFactoredMaxentWeights::checkGradientStore(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<GlobalFeatureStore>& store,
    const boost::shared_ptr<MinibatchFeatureStore>& gradient_store,
    Real eps) {
  vector<pair<int, int>> feature_indexes = store->getFeatureIndexes();
  for (const auto& index: feature_indexes) {
    store->updateFeature(index, eps);
    Real log_likelihood_plus = getLogLikelihood(corpus, indices);
    store->updateFeature(index, -eps);

    store->updateFeature(index, -eps);
    Real log_likelihood_minus = getLogLikelihood(corpus, indices);
    store->updateFeature(index, eps);

    double est_gradient = (log_likelihood_plus - log_likelihood_minus) / (2 * eps);
    if (fabs(gradient_store->getFeature(index) - est_gradient) > eps) {
      return false;
    }
  }

  return true;
}

void GlobalFactoredMaxentWeights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  throw NotImplementedException();
}

void GlobalFactoredMaxentWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient) {
  FactoredWeights::updateSquared(global_words, global_gradient);

  // The number of n-gram weights updated for each minibatch is relatively low
  // compared to the number of parameters updated in the base (factored)
  // bilinear model, so it's okay to let a single thread handle this
  // computation. Adding parallelization here is not trivial.
  #pragma omp master
  {
    U->updateSquared(global_gradient->U);
    for (size_t i = 0; i < V.size(); ++i) {
      V[i]->updateSquared(global_gradient->V[i]);
    }
  }
}

void GlobalFactoredMaxentWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient,
    const boost::shared_ptr<GlobalFactoredMaxentWeights>& adagrad) {
  FactoredWeights::updateAdaGrad(global_words, global_gradient, adagrad);

  // See comment above.
  #pragma omp master
  {
    U->updateAdaGrad(global_gradient->U, adagrad->U, config->step_size);
    for (size_t i = 0; i < V.size(); ++i) {
      V[i]->updateAdaGrad(global_gradient->V[i], adagrad->V[i], config->step_size);
    }
  }
}

Real GlobalFactoredMaxentWeights::regularizerUpdate(
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient,
    Real minibatch_factor) {
  Real ret = FactoredWeights::regularizerUpdate(
      global_gradient, minibatch_factor);

  // See comment above.
  #pragma omp master
  {
    Real sigma = minibatch_factor * config->step_size * config->l2_maxent;
    Real factor = 0.5 * minibatch_factor * config->l2_maxent;
    U->l2GradientUpdate(global_gradient->U, sigma);
    ret += U->l2Objective(global_gradient->U, factor);
    for (size_t i = 0; i < V.size(); ++i) {
      V[i]->l2GradientUpdate(global_gradient->V[i], sigma);
      ret += V[i]->l2Objective(global_gradient->V[i], factor);
    }
  }

  return ret;
}

Real GlobalFactoredMaxentWeights::getLogProb(
    int word_id, vector<int> context) const {
  int class_id = index->getClass(word_id);
  int word_class_id = index->getWordIndexInClass(word_id);
  VectorReal prediction_vector = getPredictionVector(context);

  Real class_prob = 0;
  auto ret = normalizerCache.get(context);
  if (ret.second) {
    Real class_score = U->get(context)(class_id);
    class_prob = S.col(class_id).dot(prediction_vector) + T(class_id) + class_score - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal class_probs = logSoftMax(
        S.transpose() * prediction_vector + T + U->get(context), normalizer);
    normalizerCache.set(context, normalizer);
    class_prob = class_probs(class_id);
  }

  context.insert(context.begin(), class_id);
  Real word_prob = 0;
  ret = classNormalizerCache.get(context);
  if (ret.second) {
    Real word_score = V[class_id]->get(context)(word_class_id);
    word_prob = R.col(word_id).dot(prediction_vector) + B(word_id) + word_score - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal word_probs = logSoftMax(
        classR(class_id).transpose() * prediction_vector + classB(class_id) + V[class_id]->get(context),
        normalizer);
    classNormalizerCache.set(context, normalizer);
    word_prob = word_probs(word_class_id);
  }

  return class_prob + word_prob;
}

Real GlobalFactoredMaxentWeights::getUnnormalizedScore(
    int word_id, const vector<int>& context) const {
  int class_id = index->getClass(word_id);
  int word_class_id = index->getWordIndexInClass(word_id);
  VectorReal prediction_vector = getPredictionVector(context);

  Real class_score =
      S.col(class_id).dot(prediction_vector) + T(class_id) +
      U->getValue(class_id, context);
  Real word_score =
      R.col(word_id).dot(prediction_vector) + B(word_id) +
      V[class_id]->getValue(word_class_id, context);

  return class_score + word_score;
}

bool GlobalFactoredMaxentWeights::operator==(
    const GlobalFactoredMaxentWeights& other) const {
  if (V.size() != other.V.size()) {
    return false;
  }

  for (size_t i = 0; i < V.size(); ++i) {
    if (!(V[i]->operator==(other.V[i]))) {
      return false;
    }
  }

  return FactoredWeights::operator==(other)
      && *metadata == *other.metadata
      && U->operator==(other.U);
}

} // namespace oxlm
