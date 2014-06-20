#include "lbl/factored_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/operators.h"
#include "utils/constants.h"

namespace oxlm {

FactoredWeights::FactoredWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMetadata>& metadata)
    : Weights(config, metadata), metadata(metadata),
      index(metadata->getIndex()), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();
}

FactoredWeights::FactoredWeights(
    const ModelData& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : Weights(config, metadata, training_corpus), metadata(metadata),
      index(metadata->getIndex()), S(0, 0, 0), T(0, 0), FW(0, 0) {
  allocate();

  // Initialize model weights randomly.
  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    FW(i) = gaussian(gen);
  }

  T = metadata->getClassBias();
}

void FactoredWeights::allocate() {
  int num_classes = index->getNumClasses();
  int word_width = config.word_representation_size;

  int S_size = num_classes * word_width;
  int T_size = num_classes;

  size = S_size + T_size;
  data = new Real[size];

  new (&FW) WeightsType(data, size);
  FW.setZero();

  new (&S) WordVectorsType(data, word_width, num_classes);
  new (&T) WeightsType(data + S_size, T_size);
}

FactoredWeights::~FactoredWeights() {
  delete data;
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
      corpus, indices, prediction_vectors, class_probs, word_probs);

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

  // Sigmoid derivative.
  weighted_representations.array() *= sigmoidDerivative(prediction_vectors);

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
    const boost::shared_ptr<FactoredWeights>& gradient) {
  if (!Weights::checkGradient(corpus, indices, gradient)) {
    return false;
  }

  for (int i = 0; i < size; ++i) {
    FW(i) += EPS;
    Real objective_plus = getObjective(corpus, indices);
    FW(i) -= EPS;

    FW(i) -= EPS;
    Real objective_minus = getObjective(corpus, indices);
    FW(i) += EPS;

    double est_gradient = (objective_plus - objective_minus) / (2 * EPS);
    if (fabs(gradient->FW(i) - est_gradient) > EPS) {
      return false;
    }
  }

  return true;
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
      adagrad->FW, CwiseAdagradUpdateOp<Real>(config.step_size));
}

Real FactoredWeights::regularizerUpdate(Real minibatch_factor) {
  Real ret = Weights::regularizerUpdate(minibatch_factor);

  Real sigma = minibatch_factor * config.step_size * config.l2_lbl;
  FW -= FW * sigma;
  ret += 0.5 * minibatch_factor * config.l2_lbl * FW.array().square().sum();

  return ret;
}

void FactoredWeights::clear() {
  Weights::clear();
  FW.setZero();
}

} // namespace oxlm
