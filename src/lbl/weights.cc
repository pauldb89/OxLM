#include "lbl/weights.h"

#include <random>

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

Weights::Weights(
    const ModelData& config,
    const boost::shared_ptr<Metadata>& metadata,
    bool model_weights)
    : config(config), metadata(metadata),
      Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  int num_context_words = config.vocab_size;
  int num_output_words = config.vocab_size;
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order - 1;

  int Q_size = num_context_words * word_width;
  int R_size = num_output_words * word_width;
  int C_size = config.diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  size = Q_size + R_size + context_width * C_size + B_size;
  data = new Real[size];

  new (&W) WeightsType(data, size);
  if (model_weights && config.random_weights) {
    random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0, 0.1);
    for (int i = 0; i < size; ++i) {
      W(i) = gaussian(gen);
    }
  } else {
    W.setZero();
  }

  new (&Q) WordVectorsType(data, word_width, num_context_words);
  new (&R) WordVectorsType(data + Q_size, word_width, num_output_words);
  Real* start = data + Q_size + R_size;
  for (int i = 0; i < context_width; ++i) {
    if (config.diagonal_contexts) {
      C.push_back(ContextTransformType(start, word_width, 1));
    } else {
      C.push_back(ContextTransformType(start, word_width, word_width));
    }
    start += C_size;
  }
  new (&B) WeightsType(start, B_size);

  if (model_weights) {
    cout << "===============================" << endl;
    cout << " Model parameters: " << endl;
    cout << "  Context vocab size = " << num_context_words << endl;
    cout << "  Output vocab size = " << num_output_words << endl;
    cout << "  Total parameters = " << size << endl;
    cout << "===============================" << endl;
  }
}

void Weights::getContextVectors(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors) const {
  int context_width = config.ngram_order - 1;
  int word_width = config.word_representation_size;
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

MatrixReal Weights::getPredictionVectors(
    const vector<int>& indices,
    const vector<MatrixReal>& context_vectors) const {
  int context_width = config.ngram_order - 1;
  int word_width = config.word_representation_size;
  MatrixReal prediction_vectors = MatrixReal::Zero(word_width, indices.size());

  for (int i = 0; i < context_width; ++i) {
    prediction_vectors += getContextProduct(i, context_vectors[i]);
  }

  return prediction_vectors;
}

MatrixReal Weights::getContextProduct(
    int index, const MatrixReal& representations) const {
  if (config.diagonal_contexts) {
    return C[index].asDiagonal() * representations;
  } else {
    return C[index] * representations;
  }
}

void Weights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    boost::shared_ptr<Weights>& gradient,
    Real& objective) const {
  int word_width = config.word_representation_size;
  gradient = boost::make_shared<Weights>(config, metadata);
  objective = 0;

  MatrixReal weighted_representations =
      MatrixReal::Zero(word_width, indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);

    VectorReal prediction_vector = sigmoid(prediction_vectors.col(i));
    ArrayReal word_log_probs = logSoftMax(R.transpose() * prediction_vector + B);
    VectorReal word_probs = word_log_probs.exp();

    assert(isfinite(word_log_probs(word_id)));
    objective -= word_log_probs(word_id);

    weighted_representations.col(i) = R.col(i) - R * word_probs;

    word_probs(word_id) -= 1;
    gradient->R += prediction_vector * word_probs.transpose();
    gradient->B += word_probs;
  }

  getContextGradient(
      indices, contexts, context_vectors, weighted_representations, gradient);
}

void Weights::getContextGradient(
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    boost::shared_ptr<Weights>& gradient) const {
  int context_width = config.ngram_order - 1;
  int word_width = config.word_representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, indices.size());
  for (int j = 0; j < context_width; ++j) {
    context_gradients = getContextProduct(j, weighted_representations);
    for (size_t i = 0; i < indices.size(); ++i) {
      gradient->Q.col(contexts[i][j]) += context_gradients.col(i);
    }

    if (config.diagonal_contexts) {
      gradient->C[j] = context_vectors[j] * weighted_representations.transpose();
    } else {
      gradient->C[j] = context_vectors[j].cwiseProduct(weighted_representations).rowwise().sum();
    }
  }
}

void Weights::update(const boost::shared_ptr<Weights>& gradient) {
  W += gradient->W;
}

void Weights::updateSquared(const boost::shared_ptr<Weights>& global_gradient) {
  W.array() += global_gradient->W.array().square();
}

void Weights::updateAdaGrad(
    const boost::shared_ptr<Weights>& global_gradient,
    const boost::shared_ptr<Weights>& adagrad) {
  W -= global_gradient->W.binaryExpr(
      adagrad->W, CwiseAdagradUpdateOp<Real>(config.step_size));
}

Real Weights::regularizerUpdate(Real minibatch_factor) {
  Real sigma = minibatch_factor * config.step_size * config.l2_lbl;
  W -= W * sigma;
  return 0.5 * minibatch_factor * config.l2_lbl * W.array().square().sum();
}

Real Weights::predict(int word_id, const vector<int>& context) const {
  int context_width = config.ngram_order - 1;
  int word_width = config.word_representation_size;

  VectorReal prediction_vector = VectorReal::Zero(word_width);
  for (int i = 0; i < context_width; ++i) {
    prediction_vector += getContextProduct(i, Q.col(context[i]));
  }

  VectorReal word_probs = logSoftMax(R.transpose() * prediction_vector + B);
  return word_probs(word_id);
}

void Weights::clear() {
  W.setZero();
}

Weights::~Weights() {
  delete data;
}

} // namespace oxlm
