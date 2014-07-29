#include "lbl/factored_tree_weights.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

FactoredTreeWeights::FactoredTreeWeights()
    : data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {}

FactoredTreeWeights::FactoredTreeWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<TreeMetadata>& metadata)
    : config(config), metadata(metadata), tree(metadata->getTree()),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  W.setZero();
}

FactoredTreeWeights::FactoredTreeWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<TreeMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : config(config), metadata(metadata), tree(metadata->getTree()),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();

  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    W(i) = gaussian(gen);
  }

  // Initialize bias with unigram probabilities.
  // The class biases are set to the sum of the biases in their subtree.
  VectorReal counts = VectorReal::Zero(config->vocab_size);
  for (size_t i = 0; i < training_corpus->size(); ++i) {
    counts(training_corpus->at(i)) += 1;
  }
  counts = (counts.array() + 1) / (counts.sum() + counts.size());

  B.setZero();
  for (size_t i = 0; i < config->vocab_size; ++i) {
    int node = tree->getNode(i);
    B(node) += counts(i);
    while (node != tree->getRoot()) {
      int parent = tree->getParent(node);
      B(parent) += counts(i);
      node = parent;
    }
  }
  assert(fabs(B(tree->getRoot()) - 1) < 1e-4);
  B = B.array().log();

  cout << "===============================" << endl;
  cout << " Model parameters: " << endl;
  cout << "  Context vocab size = " << config->vocab_size << endl;
  cout << "  Output vocab size = " << tree->size() << endl;
  cout << "  Total parameters = " << size << endl;
  cout << "===============================" << endl;
}

FactoredTreeWeights::FactoredTreeWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<TreeMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus,
    const vector<int>& indices)
    : config(config), metadata(metadata), tree(metadata->getTree()),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  W.setZero();
}

void FactoredTreeWeights::allocate() {
  int num_context_words = config->vocab_size;
  int num_outputs = tree->size();
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_outputs;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_outputs;

  size = Q_size + R_size + context_width * C_size + B_size;
  data = new Real[size];

  setModelParameters();
}

void FactoredTreeWeights::setModelParameters() {
  int num_context_words = config->vocab_size;
  int num_outputs = tree->size();
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_outputs;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_outputs;

  new (&W) WeightsType(data, size);

  new (&Q) WordVectorsType(data, word_width, num_context_words);
  new (&R) WordVectorsType(data + Q_size, word_width, num_outputs);

  Real* start = data + Q_size + R_size;
  for (int i = 0; i < context_width; ++i) {
    if (config->diagonal_contexts) {
      C.push_back(ContextTransformType(start, word_width, 1));
    } else {
      C.push_back(ContextTransformType(start, word_width, word_width));
    }
    start += C_size;
  }

  new (&B) WeightsType(start, B_size);
}

void FactoredTreeWeights::getContextVectors(
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

MatrixReal FactoredTreeWeights::getPredictionVectors(
    const vector<int>& indices,
    const vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  MatrixReal prediction_vectors = MatrixReal::Zero(word_width, indices.size());

  for (int i = 0; i < context_width; ++i) {
    prediction_vectors += getContextProduct(i, context_vectors[i]);
  }

  prediction_vectors = activation(config, prediction_vectors);

  return prediction_vectors;
}

MatrixReal FactoredTreeWeights::getContextProduct(
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

const Eigen::Block<const WordVectorsType> FactoredTreeWeights::classR(
    int node) const {
  vector<int> children = tree->getChildren(node);
  return R.block(0, children[0], R.rows(), children.size());
}

Eigen::Block<WordVectorsType> FactoredTreeWeights::classR(int node) {
  vector<int> children = tree->getChildren(node);
  return R.block(0, children[0], R.rows(), children.size());
}

const Eigen::VectorBlock<const WeightsType> FactoredTreeWeights::classB(
    int node) const {
  vector<int> children = tree->getChildren(node);
  return B.segment(children[0], children.size());
}

Eigen::VectorBlock<WeightsType> FactoredTreeWeights::classB(int node) {
  vector<int> children = tree->getChildren(node);
  return B.segment(children[0], children.size());
}

vector<vector<VectorReal>> FactoredTreeWeights::getProbabilities(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const MatrixReal& prediction_vectors) const {
  vector<vector<VectorReal>> probs(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    while (node != tree->getRoot()) {
      int parent = tree->getParent(node);
      VectorReal predictions = classR(parent).transpose() * prediction_vectors.col(i) + classB(parent);
      probs[i].push_back(softMax(predictions));
      node = parent;
    }
  }

  return probs;
}

MatrixReal FactoredTreeWeights::getWeightedRepresentations(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const MatrixReal& prediction_vectors,
    vector<vector<VectorReal>>& probs) const {
  int word_width = config->word_representation_size;
  MatrixReal weighted_representations =
      MatrixReal::Zero(word_width, indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    for (size_t j = 0; j < probs[i].size(); ++j) {
      int parent = tree->getParent(node);
      probs[i][j](tree->childIndex(node)) -= 1;
      weighted_representations.col(i) += classR(parent) * probs[i][j];
      node = parent;
    }
  }

  weighted_representations.array() *=
      activationDerivative(config, prediction_vectors);

  return weighted_representations;
}

void FactoredTreeWeights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  vector<vector<VectorReal>> probs;
  objective = getObjective(
      corpus, indices, contexts, context_vectors, prediction_vectors, probs);

  MatrixReal weighted_representations =
      getWeightedRepresentations(corpus, indices, prediction_vectors, probs);

  getFullGradient(
      corpus, indices, contexts, context_vectors, prediction_vectors, probs,
      weighted_representations);
}

boost::shared_ptr<FactoredTreeWeights> FactoredTreeWeights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const vector<vector<VectorReal>>& probs,
    const MatrixReal& weighted_representations) const {
  boost::shared_ptr<FactoredTreeWeights> gradient =
      boost::make_shared<FactoredTreeWeights>(config, metadata);

  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    for (size_t j = 0; j < probs[i].size(); ++j) {
      int parent = tree->getParent(node);
      gradient->classB(parent) += probs[i][j];
      gradient->classR(parent) +=
          prediction_vectors.col(i) * probs[i][j].transpose();
      node = parent;
    }
  }

  getContextGradient(
      indices, contexts, context_vectors, weighted_representations, gradient);

  return gradient;
}

void FactoredTreeWeights::getContextGradient(
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    const boost::shared_ptr<FactoredTreeWeights>& gradient) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, indices.size());
  for (int j = 0; j < context_width; ++j) {
    context_gradients = getContextProduct(j, weighted_representations, true);
    for (size_t i = 0; i < indices.size(); ++i) {
      gradient->Q.col(contexts[i][j]) += context_gradients.col(i);
    }

    if (config->diagonal_contexts) {
      gradient->C[j] = context_vectors[j].cwiseProduct(weighted_representations).rowwise().sum();
    } else {
      gradient->C[j] = weighted_representations * context_vectors[j].transpose();
    }
  }
}

Real FactoredTreeWeights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    vector<vector<VectorReal>>& probs) const {
  getContextVectors(corpus, indices, contexts, context_vectors);
  prediction_vectors = getPredictionVectors(indices, context_vectors);
  probs = getProbabilities(corpus, indices, prediction_vectors);

  Real objective = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    for (size_t j = 0; j < probs[i].size(); ++j) {
      int parent = tree->getParent(node);
      vector<int> children = tree->getChildren(parent);
      objective -= log(probs[i][j](tree->childIndex(node)));
      node = parent;
    }
  }

  return objective;
}

Real FactoredTreeWeights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  vector<vector<VectorReal>> probs;
  return getObjective(
      corpus, indices, contexts, context_vectors, prediction_vectors, probs);
}

void FactoredTreeWeights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  throw NotImplementedException();
}

bool FactoredTreeWeights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    Real eps) {
  for (int i = 0; i < W.size(); ++i) {
    W(i) += eps;
    Real objective_plus = getObjective(corpus, indices);
    W(i) -= eps;

    W(i) -= eps;
    Real objective_minus = getObjective(corpus, indices);
    W(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->W(i) - est_gradient) > eps) {
      return false;
    }
  }

  return true;
}

void FactoredTreeWeights::update(
    const boost::shared_ptr<FactoredTreeWeights>& gradient) {
  W += gradient->W;
}

void FactoredTreeWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<FactoredTreeWeights>& global_gradient) {
  W.array() += global_gradient->W.array().square();
}

void FactoredTreeWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<FactoredTreeWeights>& global_gradient,
    const boost::shared_ptr<FactoredTreeWeights>& adagrad) {
  W -= global_gradient->W.binaryExpr(
      adagrad->W, CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real FactoredTreeWeights::regularizerUpdate(
    const boost::shared_ptr<FactoredTreeWeights>& global_gradient,
    Real minibatch_factor) {
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  W -= W * sigma;
  return 0.5 * minibatch_factor * config->l2_lbl * W.array().square().sum();
}

Real FactoredTreeWeights::predict(
    int word_id, const vector<int>& context) const {
  throw NotImplementedException();
}

void FactoredTreeWeights::clearCache() {
  throw NotImplementedException();
}

MatrixReal FactoredTreeWeights::getWordVectors() const {
  throw NotImplementedException();
}

bool FactoredTreeWeights::operator==(const FactoredTreeWeights& other) const {
  return *metadata == *other.metadata
      && *tree == *other.tree
      && size == other.size
      && W.isApprox(other.W, EPS);
}

FactoredTreeWeights::~FactoredTreeWeights() {
  delete data;
}

} // namespace oxlm
