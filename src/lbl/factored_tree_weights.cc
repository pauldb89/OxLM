#include "lbl/factored_tree_weights.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

FactoredTreeWeights::FactoredTreeWeights() : Weights() {}

FactoredTreeWeights::FactoredTreeWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<TreeMetadata>& metadata)
    : Weights(config), metadata(metadata), tree(metadata->getTree()) {
  allocate();
  W.setZero();
}

FactoredTreeWeights::FactoredTreeWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<TreeMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : Weights(config), metadata(metadata), tree(metadata->getTree()) {
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
  assert(fabs(B(tree->getRoot()) - 1) < 1e-3);
  B = B.array().log();
}

void FactoredTreeWeights::printInfo() const {
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
    : Weights(config), metadata(metadata), tree(metadata->getTree()) {
  allocate();
  W.setZero();
}

void FactoredTreeWeights::allocate() {
  int num_context_words = config->vocab_size;
  int num_output_words = tree->size();
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int H_size = word_width * word_width;
  int B_size = num_output_words;

  size = Q_size + R_size + context_width * C_size +
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

void FactoredTreeWeights::setModelParameters() {
  int num_context_words = config->vocab_size;
  int num_output_words = tree->size();
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
    const vector<MatrixReal>& forward_weights) const {
  vector<vector<VectorReal>> probs(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    while (node != tree->getRoot()) {
      int parent = tree->getParent(node);
      VectorReal predictions = classR(parent).transpose() * forward_weights.back().col(i) + classB(parent);
      probs[i].push_back(softMax(predictions));
      node = parent;
    }
  }

  return probs;
}

void FactoredTreeWeights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  vector<MatrixReal> forward_weights;
  vector<vector<VectorReal>> probs;
  log_likelihood += getObjective(
      corpus, indices, contexts, context_vectors, forward_weights, probs);

  setContextWords(contexts, words);

  getFullGradient(
      corpus, indices, contexts, context_vectors, forward_weights, probs,
      gradient, words);
}

void FactoredTreeWeights::getProjectionGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<MatrixReal>& forward_weights,
    vector<vector<VectorReal>>& probs,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    MatrixReal& backward_weights,
    MinibatchWords& words) const {
  int word_width = config->word_representation_size;
  backward_weights = MatrixReal::Zero(word_width, indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    for (size_t j = 0; j < probs[i].size(); ++j) {
      words.addOutputWord(node);

      int parent = tree->getParent(node);
      probs[i][j](tree->childIndex(node)) -= 1;

      gradient->classB(parent) += probs[i][j];
      gradient->classR(parent) +=
          forward_weights.back().col(i) * probs[i][j].transpose();

      backward_weights.col(i) += classR(parent) * probs[i][j];

      node = parent;
    }
  }

  backward_weights.array() *=
      activationDerivative(config, forward_weights.back());

  propagateBackwards(forward_weights, backward_weights, gradient);
}

void FactoredTreeWeights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const vector<MatrixReal>& forward_weights,
    vector<vector<VectorReal>>& probs,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    MinibatchWords& words) const {
  MatrixReal backward_weights;
  getProjectionGradient(
      corpus, indices, forward_weights, probs, gradient, backward_weights, words);

  getContextGradient(
      indices, contexts, context_vectors, backward_weights, gradient);
}

Real FactoredTreeWeights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors,
    vector<MatrixReal>& forward_weights,
    vector<vector<VectorReal>>& probs) const {
  getContextVectors(corpus, indices, contexts, context_vectors);
  forward_weights = propagateForwards(indices, context_vectors);
  probs = getProbabilities(corpus, indices, forward_weights);

  Real log_likelihood = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    int word_id = corpus->at(indices[i]);
    int node = tree->getNode(word_id);
    for (size_t j = 0; j < probs[i].size(); ++j) {
      int parent = tree->getParent(node);
      vector<int> children = tree->getChildren(parent);
      log_likelihood -= log(probs[i][j](tree->childIndex(node)));
      node = parent;
    }
  }

  return log_likelihood;
}

Real FactoredTreeWeights::getLogLikelihood(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  vector<MatrixReal> forward_weights;
  vector<vector<VectorReal>> probs;
  return getObjective(
      corpus, indices, contexts, context_vectors, forward_weights, probs);
}

void FactoredTreeWeights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<FactoredTreeWeights>& gradient,
    Real& log_likelihood,
    MinibatchWords& words) const {
  throw NotImplementedException();
}

Real FactoredTreeWeights::getLogProb(int word_id, vector<int> context) const {
  VectorReal prediction_vector = getPredictionVector(context);

  Real log_prob = 0;
  int node = tree->getNode(word_id);
  while (node != tree->getRoot()) {
    int parent = tree->getParent(node);

    context.push_back(parent);
    auto ret = normalizerCache.get(context);
    if (ret.second) {
      log_prob += R.col(node).dot(prediction_vector) + B(node) - ret.first;
    } else {
      Real normalizer = 0;
      VectorReal log_probs = logSoftMax(
          classR(parent).transpose() * prediction_vector + classB(parent),
          normalizer);
      normalizerCache.set(context, normalizer);
      log_prob += log_probs(tree->childIndex(node));
    }
    context.pop_back();

    node = parent;
  }

  return log_prob;
}

Real FactoredTreeWeights::getUnnormalizedScore(
    int word_id, const vector<int>& context) const {
  VectorReal prediction_vector = getPredictionVector(context);

  Real score = 0;
  int node = tree->getNode(word_id);
  while (node != tree->getRoot()) {
    int parent = tree->getParent(node);
    score += R.col(node).dot(prediction_vector) + B(node);
    node = parent;
  }

  return score;
}

MatrixReal FactoredTreeWeights::getWordVectors() const {
  int word_width = config->word_representation_size;
  MatrixReal word_vectors = MatrixReal::Zero(word_width, config->vocab_size);

  // Map every word_id to its node and extract the corresponding word vector.
  for (int word_id = 0; word_id < config->vocab_size; ++word_id) {
    int node = tree->getNode(word_id);
    word_vectors.col(word_id) = R.col(node);
  }

  return word_vectors;
}

bool FactoredTreeWeights::operator==(const FactoredTreeWeights& other) const {
  return *metadata == *other.metadata
      && *tree == *other.tree
      && size == other.size
      && W.isApprox(other.W, EPS);
}

FactoredTreeWeights::~FactoredTreeWeights() {}

} // namespace oxlm
