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

FactoredWeights::FactoredWeights(const FactoredWeights& other)
    : Weights(other), metadata(other.metadata),
      index(other.index),
      data(NULL), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

size_t FactoredWeights::numParameters() const {
  return Weights::numParameters() + size;
}

void FactoredWeights::init(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& minibatch) {
  Weights::init(corpus, minibatch);
}

void FactoredWeights::allocate() {
  int num_classes = index->getNumClasses();
  int word_width = config->word_representation_size;

  int S_size = num_classes * word_width;
  int T_size = num_classes;

  size = S_size + T_size;
  data = new Real[size];

  for (int i = 0; i < config->threads; ++i) {
    mutexes.push_back(boost::make_shared<mutex>());
  }

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

void FactoredWeights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors, class_probs;
  vector<VectorReal> word_probs;
  objective += getObjective(
      corpus, indices, contexts, context_vectors, prediction_vectors,
      class_probs, word_probs);

  setContextWords(contexts, words);

  MatrixReal weighted_representations = getWeightedRepresentations(
      corpus, indices, prediction_vectors, class_probs, word_probs);

  getFullGradient(
      corpus, indices, contexts, context_vectors, prediction_vectors,
      weighted_representations, class_probs, word_probs, gradient, words);
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
    class_probs.col(i) = softMax(class_probs.col(i));
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);

    VectorReal prediction_vector = prediction_vectors.col(i);
    VectorReal word_scores = classR(class_id).transpose() * prediction_vector + classB(class_id);
    word_probs.push_back(softMax(word_scores));
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

void FactoredWeights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs,
    const boost::shared_ptr<FactoredWeights>& gradient,
    MinibatchWords& words) const {
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    class_probs(class_id, i) -= 1;
    word_probs[i](word_class_id) -= 1;
  }

  gradient->S += prediction_vectors * class_probs.transpose();
  gradient->T += class_probs.rowwise().sum();
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int class_start = index->getClassMarker(class_id);
    int class_size = index->getClassSize(class_id);

    for (int j = 0; j < class_size; ++j) {
      words.addOutputWord(class_start + j);
    }

    gradient->B.segment(class_start, class_size) += word_probs[i];
    gradient->R.block(0, class_start, gradient->R.rows(), class_size) +=
        prediction_vectors.col(i) * word_probs[i].transpose();
  }

  getContextGradient(
      indices, contexts, context_vectors, weighted_representations, gradient);
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
  if (!wordDists.get()) {
    wordDists.reset(new WordDistributions(metadata->getUnigram(), index));
  }

  VectorReal unigram = metadata->getUnigram();
  vector<vector<int>> noise_words(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int class_id = index->getClass(word_id);
    int class_start = index->getClassMarker(class_id);

    auto start_sampling = GetTime();
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_words[i].push_back(wordDists->sample(class_id));
    }
  }

  return noise_words;
}

vector<vector<int>> FactoredWeights::getNoiseClasses(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  if (!classDist.get()) {
    VectorReal class_unigram = metadata->getClassBias().array().exp();
    classDist.reset(new ClassDistribution(class_unigram));
  }

  vector<vector<int>> noise_classes(indices.size());
  auto start_sampling = GetTime();
  for (size_t i = 0; i < indices.size(); ++i) {
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_classes[i].push_back(classDist->sample());
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
    Real& objective,
    MinibatchWords& words) const {
  Weights::estimateProjectionGradient(
      corpus, indices, prediction_vectors, gradient,
      weighted_representations, objective, words);

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

void FactoredWeights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  getContextVectors(corpus, indices, contexts, context_vectors);

  setContextWords(contexts, words);

  MatrixReal prediction_vectors =
      getPredictionVectors(indices, context_vectors);

  MatrixReal weighted_representations;
  estimateProjectionGradient(
      corpus, indices, prediction_vectors, gradient,
      weighted_representations, objective, words);

  if (config->sigmoid) {
    weighted_representations.array() *= sigmoidDerivative(prediction_vectors);
  }

  getContextGradient(
      indices, contexts, context_vectors, weighted_representations, gradient);
}

void FactoredWeights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<FactoredWeights>& gradient) {
  Weights::syncUpdate(words, gradient);

  size_t block_size = FW.size() / mutexes.size() + 1;
  size_t block_start = 0;
  for (size_t i = 0; i < mutexes.size(); ++i) {
    block_size = min(block_size, FW.size() - block_start);
    lock_guard<mutex> lock(*mutexes[i]);
    FW.segment(block_start, block_size) +=
        gradient->FW.segment(block_start, block_size);
    block_start += block_size;
  }
}

Block FactoredWeights::getBlock() const {
  int thread_id = omp_get_thread_num();
  size_t block_size = FW.size() / config->threads + 1;
  size_t block_start = thread_id * block_size;
  block_size = min(block_size, FW.size() - block_start);
  return make_pair(block_start, block_size);
}

void FactoredWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<FactoredWeights>& global_gradient) {
  Weights::updateSquared(global_words, global_gradient);

  Block block = getBlock();
  FW.segment(block.first, block.second).array() +=
      global_gradient->FW.segment(block.first, block.second).array().square();
}

void FactoredWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<FactoredWeights>& global_gradient,
    const boost::shared_ptr<FactoredWeights>& adagrad) {
  Weights::updateAdaGrad(global_words, global_gradient, adagrad);

  Block block = getBlock();
  FW.segment(block.first, block.second) -=
      global_gradient->FW.segment(block.first, block.second).binaryExpr(
          adagrad->FW.segment(block.first, block.second),
          CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real FactoredWeights::regularizerUpdate(
    const boost::shared_ptr<FactoredWeights>& global_gradient,
    Real minibatch_factor) {
  Real ret = Weights::regularizerUpdate(global_gradient, minibatch_factor);

  Block block = getBlock();
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  FW.segment(block.first, block.second) -=
      FW.segment(block.first, block.second) * sigma;

  Real squares = FW.segment(block.first, block.second).array().square().sum();
  ret += 0.5 * minibatch_factor * config->l2_lbl * squares;

  return ret;
}

void FactoredWeights::clear(const MinibatchWords& words, bool parallel_update) {
  Weights::clear(words, parallel_update);

  if (parallel_update) {
    Block block = getBlock();
    FW.segment(block.first, block.second).setZero();
  } else {
    FW.setZero();
  }
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
