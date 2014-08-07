#include "lbl/factored_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/operators.h"

namespace oxlm {

FactoredWeights::FactoredWeights()
    : data(NULL), S(0, 0, 0), T(0, 0), FW(0, 0) {}

FactoredWeights::FactoredWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMetadata>& metadata)
    : Weights(config, metadata), metadata(metadata),
      index(metadata->getIndex()),
      data(NULL), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();
  FW.setZero();
}

FactoredWeights::FactoredWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : Weights(config, metadata, training_corpus), metadata(metadata),
      index(metadata->getIndex()),
      data(NULL), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();

  // Initialize model weights randomly.
  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    FW(i) = gaussian(gen);
  }

  T = metadata->getClassBias();
}

FactoredWeights::FactoredWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices)
    : Weights(config, metadata, corpus, indices), metadata(metadata),
      index(metadata->getIndex()),
      data(NULL), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();
  FW.setZero();
}

FactoredWeights::FactoredWeights(const FactoredWeights& other)
    : Weights(other), metadata(other.metadata),
      index(other.index),
      data(NULL), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

void FactoredWeights::allocate() {
  int num_classes = index->getNumClasses();
  int word_width = config->word_representation_size;

  int S_size = num_classes * word_width;
  int T_size = num_classes;

  size = S_size + T_size;
  data = new Real[size];

  setModelParameters();
}

void FactoredWeights::setModelParameters() {
  int num_classes = index->getNumClasses();
  int word_width = config->word_representation_size;

  int S_size = num_classes * word_width;
  int T_size = num_classes;

  new (&FW) WeightsType(data, size);

  new (&S) WordVectorsType(data, word_width, num_classes);
  new (&T) WeightsType(data + S_size, T_size);
}

Real FactoredWeights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal class_probs;
  vector<VectorReal> word_probs;
  return getObjective(
      corpus, indices, contexts, context_vectors, prediction_vectors,
      class_probs, word_probs);
}

Real FactoredWeights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  getContextVectors(corpus, indices, contexts, context_vectors);
  prediction_vectors = getPredictionVectors(indices, context_vectors);
  getProbabilities(
      corpus, indices, contexts, prediction_vectors, class_probs, word_probs);

  Real objective = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);

    objective -= log(class_probs(class_id, i));
    objective -= log(word_probs[i](word_class_id));
  }

  return objective;
}

boost::shared_ptr<FactoredWeights> FactoredWeights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    Real& objective) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors, class_probs;
  vector<VectorReal> word_probs;
  objective = getObjective(
      corpus, indices, contexts, context_vectors, prediction_vectors,
      class_probs, word_probs);

  MatrixReal weighted_representations = getWeightedRepresentations(
      corpus, indices, prediction_vectors, class_probs, word_probs);

  return getFullGradient(
      corpus, indices, contexts, context_vectors, prediction_vectors,
      weighted_representations, class_probs, word_probs);
}

MatrixReal FactoredWeights::classR(int class_id) const {
  int class_start = index->getClassMarker(class_id);
  int class_size = index->getClassSize(class_id);
  return R.block(0, class_start, R.rows(), class_size);
}

VectorReal FactoredWeights::classB(int class_id) const {
  int class_start = index->getClassMarker(class_id);
  int class_size = index->getClassSize(class_id);
  return B.segment(class_start, class_size);
}

void FactoredWeights::getProbabilities(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const MatrixReal& prediction_vectors,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  class_probs = S.transpose() * prediction_vectors + T * MatrixReal::Ones(1, indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);

    VectorReal prediction_vector = prediction_vectors.col(i);
    VectorReal word_scores = classR(class_id).transpose() * prediction_vector + classB(class_id);
    word_probs.push_back(softMax(word_scores));

    class_probs.col(i) = softMax(class_probs.col(i));
  }
}

MatrixReal FactoredWeights::getWeightedRepresentations(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const MatrixReal& prediction_vectors,
    const MatrixReal& class_probs,
    const vector<VectorReal>& word_probs) const {
  MatrixReal weighted_representations = S * class_probs;

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);

    weighted_representations.col(i) += classR(class_id) * word_probs[i];
    weighted_representations.col(i) -= S.col(class_id) + R.col(word_id);
  }

  if (config->sigmoid) {
    weighted_representations.array() *= sigmoidDerivative(prediction_vectors);
  }

  return weighted_representations;
}

boost::shared_ptr<FactoredWeights> FactoredWeights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  boost::shared_ptr<FactoredWeights> gradient =
      boost::make_shared<FactoredWeights>(config, metadata);

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    class_probs(class_id, i) -= 1;
    word_probs[i](word_class_id) -= 1;
  }

  gradient->S = prediction_vectors * class_probs.transpose();
  gradient->T = class_probs.rowwise().sum();
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int class_start = index->getClassMarker(class_id);
    int class_size = index->getClassSize(class_id);

    gradient->B.segment(class_start, class_size) += word_probs[i];
    gradient->R.block(0, class_start, gradient->R.rows(), class_size) +=
        prediction_vectors.col(i) * word_probs[i].transpose();
  }

  getContextGradient(
      indices, contexts, context_vectors, weighted_representations, gradient);

  return gradient;
}

bool FactoredWeights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredWeights>& gradient,
    double eps) {
  if (!Weights::checkGradient(corpus, indices, gradient, eps)) {
    return false;
  }

  for (int i = 0; i < size; ++i) {
    FW(i) += eps;
    Real objective_plus = getObjective(corpus, indices);
    FW(i) -= eps;

    FW(i) -= eps;
    Real objective_minus = getObjective(corpus, indices);
    FW(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->FW(i) - est_gradient) > eps) {
      return false;
    }
  }

  return true;
}

vector<vector<int>> FactoredWeights::getNoiseWords(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  random_device rd;
  mt19937 gen(rd());
  VectorReal unigram = metadata->getUnigram();
  vector<vector<int>> noise_words(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int class_start = index->getClassMarker(class_id);
    int class_size = index->getClassSize(class_id);

    discrete_distribution<int> discrete(
        unigram.data() + class_start,
        unigram.data() + class_start + class_size);
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_words[i].push_back(class_start + discrete(gen));
    }
  }

  return noise_words;
}

vector<vector<int>> FactoredWeights::getNoiseClasses(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  random_device rd;
  mt19937 gen(rd());
  VectorReal class_unigram = metadata->getClassBias().array().exp();
  discrete_distribution<int> discrete(
      class_unigram.data(), class_unigram.data() + class_unigram.size());
  vector<vector<int>> noise_classes(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_classes[i].push_back(discrete(gen));
    }
  }

  return noise_classes;
}

void FactoredWeights::estimateProjectionGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const MatrixReal& prediction_vectors,
    const boost::shared_ptr<FactoredWeights>& gradient,
    MatrixReal& weighted_representations,
    Real& objective) const {
  Weights::estimateProjectionGradient(
      corpus, indices, prediction_vectors, gradient,
      weighted_representations, objective);

  int noise_samples = config->noise_samples;
  VectorReal class_unigram = metadata->getClassBias().array().exp();
  vector<vector<int>> noise_classes = getNoiseClasses(corpus, indices);
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    Real log_pos_prob = S.col(class_id).dot(prediction_vectors.col(i)) + T(class_id);
    Real pos_prob = exp(log_pos_prob);
    assert(pos_prob <= numeric_limits<Real>::max());

    Real pos_weight = (noise_samples * class_unigram(class_id)) / (pos_prob + noise_samples * class_unigram(class_id));
    weighted_representations.col(i) -= pos_weight * S.col(class_id);

    objective -= log(1 - pos_weight);

    gradient->S.col(class_id) -= pos_weight * prediction_vectors.col(i);
    gradient->T(class_id) -= pos_weight;

    for (int j = 0; j < noise_samples; ++j) {
      int noise_class_id = noise_classes[i][j];
      Real log_neg_prob = S.col(noise_class_id).dot(prediction_vectors.col(i)) + T(noise_class_id);
      Real neg_prob = exp(log_neg_prob);
      assert(neg_prob <= numeric_limits<Real>::max());

      Real neg_weight = neg_prob / (neg_prob + noise_samples * class_unigram(noise_class_id));
      weighted_representations.col(i) += neg_weight * S.col(noise_class_id);

      objective -= log(1 - neg_weight);

      gradient->S.col(noise_class_id) += neg_weight * prediction_vectors.col(i);
      gradient->T(noise_class_id) += neg_weight;
    }
  }
}

boost::shared_ptr<FactoredWeights> FactoredWeights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    Real& objective) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  getContextVectors(corpus, indices, contexts, context_vectors);

  MatrixReal prediction_vectors =
      getPredictionVectors(indices, context_vectors);

  boost::shared_ptr<FactoredWeights> gradient =
      boost::make_shared<FactoredWeights>(config, metadata);
  MatrixReal weighted_representations;
  estimateProjectionGradient(
      corpus, indices, prediction_vectors, gradient,
      weighted_representations, objective);

  if (config->sigmoid) {
    weighted_representations.array() *= sigmoidDerivative(prediction_vectors);
  }

  getContextGradient(
      indices, contexts, context_vectors, weighted_representations, gradient);

  return gradient;
}

void FactoredWeights::update(
    const boost::shared_ptr<FactoredWeights>& gradient) {
  Weights::update(gradient);
  FW += gradient->FW;
}

void FactoredWeights::updateSquared(
    const boost::shared_ptr<FactoredWeights>& global_gradient) {
  Weights::updateSquared(global_gradient);
  FW.array() += global_gradient->FW.array().square();
}

void FactoredWeights::updateAdaGrad(
    const boost::shared_ptr<FactoredWeights>& global_gradient,
    const boost::shared_ptr<FactoredWeights>& adagrad) {
  Weights::updateAdaGrad(global_gradient, adagrad);
  FW -= global_gradient->FW.binaryExpr(
      adagrad->FW, CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real FactoredWeights::regularizerUpdate(
    const boost::shared_ptr<FactoredWeights>& global_gradient,
    Real minibatch_factor) {
  Real ret = Weights::regularizerUpdate(global_gradient, minibatch_factor);

  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  FW -= FW * sigma;
  ret += 0.5 * minibatch_factor * config->l2_lbl * FW.array().square().sum();

  return ret;
}

Real FactoredWeights::predict(int word_id, vector<int> context) const {
  int class_id = index->getClass(word_id);
  int word_class_id = index->getWordIndexInClass(word_id);
  VectorReal prediction_vector = getPredictionVector(context);

  Real class_prob;
  auto ret = normalizerCache.get(context);
  if (ret.second) {
    class_prob = S.col(class_id).dot(prediction_vector) + T(class_id) - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal class_probs = logSoftMax(
        S.transpose() * prediction_vector + T, normalizer);
    normalizerCache.set(context, normalizer);
    class_prob = class_probs(class_id);
  }

  context.insert(context.begin(), class_id);
  Real word_prob;
  ret = classNormalizerCache.get(context);
  if (ret.second) {
    word_prob = R.col(word_id).dot(prediction_vector) + B(word_id) - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal word_probs = logSoftMax(
        classR(class_id).transpose() * prediction_vector + classB(class_id),
        normalizer);
    classNormalizerCache.set(context, normalizer);
    word_prob = word_probs(word_class_id);
  }

  return class_prob + word_prob;
}

void FactoredWeights::clearCache() {
  Weights::clearCache();
  classNormalizerCache.clear();
}

bool FactoredWeights::operator==(const FactoredWeights& other) const {
  return Weights::operator==(other)
      && *metadata == *other.metadata
      && *index == *other.index
      && size == other.size
      && FW == other.FW;
}

FactoredWeights::~FactoredWeights() {
  delete data;
}



} // namespace oxlm
