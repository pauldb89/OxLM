#include "lbl/weights.h"

#include <iomanip>
#include <random>

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

Weights::Weights() : data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {}

Weights::Weights(
    const boost::shared_ptr<ModelData>& config, const boost::shared_ptr<Metadata>& metadata)
    : config(config), metadata(metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  W.setZero();
}

Weights::Weights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Metadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : config(config), metadata(metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();

  // Initialize model weights randomly.
  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    W(i) = gaussian(gen);
  }

  // Initialize bias with unigram probabilities.
  VectorReal counts = VectorReal::Zero(config->vocab_size);
  for (size_t i = 0; i < training_corpus->size(); ++i) {
    counts(training_corpus->at(i)) += 1;
  }
  B = ((counts.array() + 1) / (counts.sum() + counts.size())).log();
}

Weights::Weights(const Weights& other)
    : config(other.config), metadata(other.metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

void Weights::allocate() {
  int num_context_words = config->vocab_size;
  int num_output_words = config->vocab_size;
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int H_size = word_width * word_width;
  int B_size = num_output_words;

  size = Q_size + R_size + context_width * C_size
       + config->hidden_layers * H_size + B_size;
  data = new Real[size];

  for (int i = 0; i < num_context_words; ++i) {
    mutexesQ.push_back(boost::make_shared<mutex>());
  }
  for (int i = 0; i < num_output_words; ++i) {
    mutexesR.push_back(boost::make_shared<mutex>());
  }
  for (int i = 0; i < config->hidden_layers; ++i) {
    mutexesH.push_back(boost::make_shared<mutex>());
  }
  for (int i = 0; i < context_width; ++i) {
    mutexesC.push_back(boost::make_shared<mutex>());
  }
  mutexB = boost::make_shared<mutex>();

  setModelParameters();
}

void Weights::setModelParameters() {
  int num_context_words = config->vocab_size;
  int num_output_words = config->vocab_size;
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int H_size = word_width * word_width;
  int B_size = num_output_words;

  new (&W) WeightsType(data, size);

  new (&Q) WordVectorsType(data, word_width, num_context_words);
  new (&R) WordVectorsType(data + Q_size, word_width, num_output_words);

  Real* start = data + Q_size + R_size;
  for (int i = 0; i < context_width; ++i) {
    if (config->diagonal_contexts) {
      C.push_back(ContextTransformType(start, word_width, 1));
    } else {
      C.push_back(ContextTransformType(start, word_width, word_width));
    }
    start += C_size;
  }

  for (int i = 0; i < config->hidden_layers; ++i) {
    H.push_back(HiddenLayer(start, word_width, word_width));
    start += H_size;
  }

  new (&B) WeightsType(start, B_size);
}

size_t Weights::numParameters() const {
  return size;
}

void Weights::printInfo() const {
  cout << "===============================" << endl;
  cout << " Model parameters: " << endl;
  cout << "  Context vocab size = " << config->vocab_size << endl;
  cout << "  Output vocab size = " << config->vocab_size << endl;
  cout << "  Total parameters = " << numParameters() << endl;
  cout << "===============================" << endl;
}

void Weights::init(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& minibatch) {}

void Weights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<Weights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  vector<MatrixReal> forward_weights;
  MatrixReal word_probs;
  log_likelihood += getObjective(
      corpus, indices, contexts, context_vectors, forward_weights, word_probs);

  setContextWords(contexts, words);

  getFullGradient(
      corpus, indices, contexts, context_vectors, forward_weights,
      word_probs, gradient, words);
}

void Weights::getContextVectors(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, context_width);

  contexts.resize(indices.size());
  context_vectors.resize(
      context_width, MatrixReal::Zero(word_width, indices.size()));
  for (size_t i = 0; i < indices.size(); ++i) {
    contexts[i] = processor->extract(indices[i]);
    for (int j = 0; j < context_width; ++j) {
      context_vectors[j].col(i) = Q.col(contexts[i][j]);
    }
  }
}

void Weights::setContextWords(
    const vector<vector<int>>& contexts,
    MinibatchWords& words) const {
  for (const auto& context: contexts) {
    for (int word_id: context) {
      words.addContextWord(word_id);
    }
  }
}

MatrixReal Weights::getPredictionVectors(
    const vector<int>& indices,
    const vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;

  MatrixReal prediction_vectors = MatrixReal::Zero(word_width, indices.size());
  for (int i = 0; i < context_width; ++i) {
    prediction_vectors += getContextProduct(i, context_vectors[i]);
  }

  return activation<MatrixReal>(config, prediction_vectors);
}

vector<MatrixReal> Weights::propagateForwards(
    const vector<int>& indices,
    const vector<MatrixReal>& context_vectors) const {
  int word_width = config->word_representation_size;
  vector<MatrixReal> forward_weights(
      config->hidden_layers + 1, MatrixReal::Zero(word_width, indices.size()));

  forward_weights[0] = getPredictionVectors(indices, context_vectors);
  for (size_t i = 0; i < config->hidden_layers; ++i) {
    forward_weights[i + 1] =
        activation<MatrixReal>(config, H[i] * forward_weights[i]);
  }

  return forward_weights;
}

MatrixReal Weights::getContextProduct(
    int index, const MatrixReal& representations, bool transpose) const {
  if (config->diagonal_contexts) {
    return C[index].asDiagonal() * representations;
  } else {
    if (transpose) {
      return C[index].transpose() * representations;
    } else {
      return C[index] * representations;
    }
  }
}

MatrixReal Weights::getProbabilities(
    const vector<int>& indices,
    const vector<MatrixReal>& forward_weights) const {
  MatrixReal word_probs = R.transpose() * forward_weights.back() + B * MatrixReal::Ones(1, indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    word_probs.col(i) = softMax(word_probs.col(i));
  }

  return word_probs;
}

void Weights::propagateBackwards(
    const vector<MatrixReal>& forward_weights,
    MatrixReal& backward_weights,
    const boost::shared_ptr<Weights>& gradient) const {
  for (int i = config->hidden_layers - 1; i >= 0; --i) {
    gradient->H[i] += backward_weights * forward_weights[i].transpose();

    backward_weights = H[i].transpose() * backward_weights;
    backward_weights.array() *=
        activationDerivative(config, forward_weights[i]);
  }
}

void Weights::getProjectionGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<MatrixReal>& forward_weights,
    const MatrixReal& word_probs,
    const boost::shared_ptr<Weights>& gradient,
    MatrixReal& backward_weights,
    MinibatchWords& words) const {
  for (size_t word_id = 0; word_id < config->vocab_size; ++word_id) {
    words.addOutputWord(word_id);
  }

  backward_weights = word_probs;
  for (size_t i = 0; i < indices.size(); ++i) {
    backward_weights(corpus->at(indices[i]), i) -= 1;
  }

  gradient->R += forward_weights.back() * backward_weights.transpose();
  gradient->B += backward_weights.rowwise().sum();

  backward_weights = R * backward_weights;
  backward_weights.array() *=
      activationDerivative(config, forward_weights.back());
}

void Weights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const vector<MatrixReal>& forward_weights,
    const MatrixReal& word_probs,
    const boost::shared_ptr<Weights>& gradient,
    MinibatchWords& words) const {
  MatrixReal backward_weights;
  getProjectionGradient(
      corpus, indices, forward_weights, word_probs, gradient,
      backward_weights, words);

  propagateBackwards(forward_weights, backward_weights, gradient);

  getContextGradient(
      indices, contexts, context_vectors, backward_weights, gradient);
}

void Weights::getContextGradient(
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& backward_weights,
    const boost::shared_ptr<Weights>& gradient) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, indices.size());
  for (int j = 0; j < context_width; ++j) {
    context_gradients = getContextProduct(j, backward_weights, true);
    for (size_t i = 0; i < indices.size(); ++i) {
      gradient->Q.col(contexts[i][j]) += context_gradients.col(i);
    }

    if (config->diagonal_contexts) {
      gradient->C[j] += context_vectors[j].cwiseProduct(backward_weights).rowwise().sum();
    } else {
      gradient->C[j] += backward_weights * context_vectors[j].transpose();
    }
  }
}

bool Weights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<Weights>& gradient,
    double eps) {
  for (int i = 0; i < size; ++i) {
    W(i) += eps;
    Real log_likelihood_plus = getLogLikelihood(corpus, indices);
    W(i) -= eps;

    W(i) -= eps;
    Real log_likelihood_minus = getLogLikelihood(corpus, indices);
    W(i) += eps;

    double est_gradient = (log_likelihood_plus - log_likelihood_minus) / (2 * eps);
    if (fabs(gradient->W(i) - est_gradient) > eps) {
      cout << i << " " << gradient->W(i) << " " << est_gradient << endl;
      return false;
    }
  }

  return true;
}

Real Weights::getLogLikelihood(
    const boost::shared_ptr<Corpus>& corpus, const vector<int>& indices) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  vector<MatrixReal> forward_weights;
  MatrixReal word_probs;
  return getObjective(
      corpus, indices, contexts, context_vectors, forward_weights, word_probs);
}

Real Weights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors,
    vector<MatrixReal>& forward_weights,
    MatrixReal& word_probs) const {
  getContextVectors(corpus, indices, contexts, context_vectors);
  forward_weights = propagateForwards(indices, context_vectors);
  word_probs = getProbabilities(indices, forward_weights);

  Real log_likelihood = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    log_likelihood -= log(word_probs(corpus->at(indices[i]), i));
  }

  return log_likelihood;
}

vector<vector<int>> Weights::getNoiseWords(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  vector<vector<int>> noise_words(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_words[i].push_back(corpus->at(rand() % corpus->size()));
    }
  }

  return noise_words;
}

void Weights::estimateProjectionGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<MatrixReal>& forward_weights,
    const boost::shared_ptr<Weights>& gradient,
    MatrixReal& backward_weights,
    Real& log_likelihood,
    MinibatchWords& words) const {
  int noise_samples = config->noise_samples;
  int word_width = config->word_representation_size;
  VectorReal unigram = metadata->getUnigram();
  vector<vector<int>> noise_words = getNoiseWords(corpus, indices);

  for (size_t i = 0; i < indices.size(); ++i) {
    words.addOutputWord(corpus->at(indices[i]));
    for (int word_id: noise_words[i]) {
      words.addOutputWord(word_id);
    }
  }

  Real log_num_samples = log(noise_samples);
  backward_weights = MatrixReal::Zero(word_width, indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    Real log_score = R.col(word_id).dot(forward_weights.back().col(i)) + B(word_id);
    Real log_noise = log_num_samples + log(unigram(word_id));
    Real log_norm = LogAdd(log_score, log_noise);

    log_likelihood -= log_score - log_norm;

    Real prob = exp(log_noise - log_norm);
    assert(prob <= numeric_limits<Real>::max());
    backward_weights.col(i) -= prob * R.col(word_id);

    gradient->R.col(word_id) -= prob * forward_weights.back().col(i);
    gradient->B(word_id) -= prob;

    for (int j = 0; j < noise_samples; ++j) {
      int noise_word_id = noise_words[i][j];
      Real log_score = R.col(noise_word_id).dot(forward_weights.back().col(i)) + B(noise_word_id);
      Real log_noise = log_num_samples + log(unigram(noise_word_id));
      Real log_norm = LogAdd(log_score, log_noise);

      log_likelihood -= log_noise - log_norm;

      Real prob = exp(log_score - log_norm);
      assert(prob <= numeric_limits<Real>::max());
      backward_weights.col(i) += prob * R.col(noise_word_id);

      gradient->R.col(noise_word_id) += prob * forward_weights.back().col(i);
      gradient->B(noise_word_id) += prob;
    }
  }
}

void Weights::estimateFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const vector<MatrixReal>& forward_weights,
    const boost::shared_ptr<Weights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  MatrixReal backward_weights;
  estimateProjectionGradient(
      corpus, indices, forward_weights, gradient,
      backward_weights, log_likelihood, words);

  // In FactoredWeights, the backward_weights add up contributions from both
  // word and class predictions. If the following non-linearity derivative was
  // applied in estimateProjectionGradient, that method would have to be
  // reimplemented in FactoredWeights, resulting in a lot of code duplication.
  backward_weights.array() *=
      activationDerivative(config, forward_weights.back());
  propagateBackwards(forward_weights, backward_weights, gradient);

  getContextGradient(
      indices, contexts, context_vectors, backward_weights, gradient);
}

void Weights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<Weights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  getContextVectors(corpus, indices, contexts, context_vectors);

  setContextWords(contexts, words);

  vector<MatrixReal> forward_weights =
      propagateForwards(indices, context_vectors);

  estimateFullGradient(
      corpus, indices, contexts, context_vectors, forward_weights, gradient,
      log_likelihood, words);
}

void Weights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<Weights>& gradient) {
  for (int word_id: words.getContextWordsSet()) {
    lock_guard<mutex> lock(*mutexesQ[word_id]);
    Q.col(word_id) += gradient->Q.col(word_id);
  }

  for (int word_id: words.getOutputWordsSet()) {
    lock_guard<mutex> lock(*mutexesR[word_id]);
    R.col(word_id) += gradient->R.col(word_id);
  }

  for (size_t i = 0; i < C.size(); ++i) {
    lock_guard<mutex> lock(*mutexesC[i]);
    C[i] += gradient->C[i];
  }

  for (size_t i = 0; i < H.size(); ++i) {
    lock_guard<mutex> lock(*mutexesH[i]);
    H[i] += gradient->H[i];
  }

  lock_guard<mutex> lock(*mutexB);
  B += gradient->B;
}

Block Weights::getBlock(int start, int size) const {
  int thread_id = omp_get_thread_num();
  size_t block_size = size / config->threads + 1;
  size_t block_start = start + thread_id * block_size;
  block_size = min(block_size, start + size - block_start);

  return make_pair(block_start, block_size);
}

void Weights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient) {
  for (int word_id: global_words.getContextWords()) {
    Q.col(word_id).array() += global_gradient->Q.col(word_id).array().square();
  }

  for (int word_id: global_words.getOutputWords()) {
    R.col(word_id).array() += global_gradient->R.col(word_id).array().square();
  }

  Block block = getBlock(Q.size() + R.size(), W.size() - (Q.size() + R.size()));
  W.segment(block.first, block.second).array() +=
      global_gradient->W.segment(block.first, block.second).array().square();
}

void Weights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient,
    const boost::shared_ptr<Weights>& adagrad) {
  for (int word_id: global_words.getContextWords()) {
    Q.col(word_id) -= global_gradient->Q.col(word_id).binaryExpr(
        adagrad->Q.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  for (int word_id: global_words.getOutputWords()) {
    R.col(word_id) -= global_gradient->R.col(word_id).binaryExpr(
        adagrad->R.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  Block block = getBlock(Q.size() + R.size(), W.size() - (Q.size() + R.size()));
  W.segment(block.first, block.second) -=
      global_gradient->W.segment(block.first, block.second).binaryExpr(
          adagrad->W.segment(block.first, block.second),
          CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real Weights::regularizerUpdate(
    const boost::shared_ptr<Weights>& global_gradient,
    Real minibatch_factor) {
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  Block block = getBlock(0, W.size());
  W.segment(block.first, block.second) -=
      W.segment(block.first, block.second) * sigma;

  Real sum = W.segment(block.first, block.second).array().square().sum();
  return 0.5 * minibatch_factor * config->l2_lbl * sum;
}

void Weights::clear(const MinibatchWords& words, bool parallel_update) {
  if (parallel_update) {
    for (int word_id: words.getContextWords()) {
      Q.col(word_id).setZero();
    }

    for (int word_id: words.getOutputWords()) {
      R.col(word_id).setZero();
    }

    Block block = getBlock(Q.size() + R.size(), W.size() - (Q.size() + R.size()));
    W.segment(block.first, block.second).setZero();
  } else {
    for (int word_id: words.getContextWordsSet()) {
      Q.col(word_id).setZero();
    }

    for (int word_id: words.getOutputWordsSet()) {
      R.col(word_id).setZero();
    }

    W.segment(Q.size() + R.size(), W.size() - (Q.size() + R.size())).setZero();
  }
}

VectorReal Weights::getPredictionVector(const vector<int>& context) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;

  VectorReal prediction_vector = VectorReal::Zero(word_width);
  for (int i = 0; i < context_width; ++i) {
    if (config->diagonal_contexts) {
      prediction_vector += C[i].asDiagonal() * Q.col(context[i]);
    } else {
      prediction_vector += C[i] * Q.col(context[i]);
    }
  }

  prediction_vector = activation(config, prediction_vector);

  for (int j = 0; j < config->hidden_layers; ++j) {
    prediction_vector =
        activation<VectorReal>(config, H[j] * prediction_vector);
  }

  return prediction_vector;
}

Real Weights::getLogProb(int word_id, vector<int> context) const {
  VectorReal prediction_vector = getPredictionVector(context);

  auto ret = normalizerCache.get(context);
  if (ret.second) {
    return R.col(word_id).dot(prediction_vector) + B(word_id) - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal word_probs = logSoftMax(
        R.transpose() * prediction_vector + B, normalizer);
    normalizerCache.set(context, normalizer);
    return word_probs(word_id);
  }
}

Real Weights::getUnnormalizedScore(
    int word_id, const vector<int>& context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  return R.col(word_id).dot(prediction_vector) + B(word_id);
}

void Weights::clearCache() {
  normalizerCache.clear();
}

MatrixReal Weights::getWordVectors() const {
  return R;
}

bool Weights::operator==(const Weights& other) const {
  return *config == *other.config
      && *metadata == *other.metadata
      && size == other.size
      && W == other.W;
}

Weights::~Weights() {
  delete data;
}

} // namespace oxlm
