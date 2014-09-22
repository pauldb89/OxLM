#include "lbl/source_factored_weights.h"

#include "lbl/operators.h"
#include "lbl/parallel_processor.h"

namespace oxlm {

SourceFactoredWeights::SourceFactoredWeights()
    : data(NULL), size(0), SQ(0, 0, 0), SW(0, 0) {}

SourceFactoredWeights::SourceFactoredWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMetadata>& metadata)
    : FactoredWeights(config, metadata),
      data(NULL), size(0), SQ(0, 0, 0), SW(0, 0) {
  allocate();
  SW.setZero();
}

SourceFactoredWeights::SourceFactoredWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : FactoredWeights(config, metadata, training_corpus),
      data(NULL), size(0), SQ(0, 0, 0), SW(0, 0) {
  allocate();

  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    SW(i) = gaussian(gen);
  }
}

SourceFactoredWeights::SourceFactoredWeights(
    const SourceFactoredWeights& other)
    : FactoredWeights(other),
      data(NULL), size(0), SQ(0, 0, 0), SW(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

void SourceFactoredWeights::allocate() {
  int word_width = config->word_representation_size;
  int num_source_words = config->source_vocab_size;
  int source_context_width = 2 * config->source_order - 1;

  int SQ_size = word_width * num_source_words;
  int SC_size = config->diagonal_contexts ? word_width : word_width * word_width;

  size = SQ_size + source_context_width * SC_size;
  data = new Real[size];

  for (int i = 0; i < num_source_words; ++i) {
    mutexesSQ.push_back(boost::make_shared<mutex>());
  }

  for (int i = 0; i < source_context_width; ++i) {
    mutexesSC.push_back(boost::make_shared<mutex>());
  }

  setModelParameters();
}

void SourceFactoredWeights::setModelParameters() {
  int word_width = config->word_representation_size;
  int num_source_words = config->source_vocab_size;
  int source_context_width = 2 * config->source_order - 1;

  int SQ_size = word_width * num_source_words;
  int SC_size = config->diagonal_contexts ? word_width : word_width * word_width;

  new (&SW) WeightsType(data, size);

  new (&SQ) WordVectorsType(data, word_width, num_source_words);

  Real* start = data + SQ_size;
  for (int i = 0; i < source_context_width; ++i) {
    if (config->diagonal_contexts) {
      SC.push_back(ContextTransformType(start, word_width, 1));
    } else {
      SC.push_back(ContextTransformType(start, word_width, word_width));
    }
    start += SC_size;
  }
}

size_t SourceFactoredWeights::numParameters() const {
  return FactoredWeights::numParameters() + size;
}

void SourceFactoredWeights::printInfo() const {
  cout << "===============================" << endl;
  cout << " Model parameters: " << endl;
  cout << "  Source context vocab size = " << config->source_vocab_size << endl;
  cout << "  Target context vocab size = " << config->vocab_size << endl;
  cout << "  Output vocab size = " << config->vocab_size << endl;
  cout << "  Total parameters = " << numParameters() << endl;
  cout << "===============================" << endl;
}

void SourceFactoredWeights::init(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& minibatch) {
  FactoredWeights::init(corpus, minibatch);
}

void SourceFactoredWeights::getContextVectors(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors) const {
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;
  int source_context_width = 2 * config->source_order - 1;
  int total_width = context_width + source_context_width;

  boost::shared_ptr<ParallelProcessor> processor =
      boost::make_shared<ParallelProcessor>(
          corpus, context_width, source_context_width);

  contexts.resize(indices.size());
  context_vectors.resize(
      total_width, MatrixReal::Zero(word_width, indices.size()));
  for (size_t i = 0; i < indices.size(); ++i) {
    contexts[i] = processor->extract(indices[i]);
    assert(contexts[i].size() == total_width);
    for (int j = 0; j < context_width; ++j) {
      context_vectors[j].col(i) = Q.col(contexts[i][j]);
    }
    for (int j = 0; j < source_context_width; ++j) {
      context_vectors[context_width + j].col(i) =
          SQ.col(contexts[i][context_width + j]);
    }
  }
}

void SourceFactoredWeights::setContextWords(
    const vector<vector<int>>& contexts,
    MinibatchWords& words) const {
  int context_width = config->ngram_order - 1;
  int source_context_width = 2 * config->source_order - 1;
  for (const auto& context: contexts) {
    for (int i = 0; i < context_width; ++i) {
      words.addContextWord(context[i]);
    }

    for (int i = 0; i < source_context_width; ++i) {
      words.addSourceWord(context[context_width + i]);
    }
  }
}

MatrixReal SourceFactoredWeights::getPredictionVectors(
    const vector<int>& indices,
    const vector<MatrixReal>& context_vectors) const {
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;
  int source_context_width = 2 * config->source_order - 1;

  MatrixReal prediction_vectors = MatrixReal::Zero(word_width, indices.size());
  for (int i = 0; i < context_width; ++i) {
    prediction_vectors += getContextProduct(i, context_vectors[i]);
  }

  for (int i = 0; i < source_context_width; ++i) {
    prediction_vectors +=
        getSourceContextProduct(i, context_vectors[context_width + i]);
  }

  if (config->sigmoid) {
    for (size_t i = 0; i < indices.size(); ++i) {
      prediction_vectors.col(i) = sigmoid(prediction_vectors.col(i));
    }
  }

  return prediction_vectors;
}

void SourceFactoredWeights::getContextGradient(
    const vector<int>& indices,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    const boost::shared_ptr<Weights>& base_gradient) const {
  FactoredWeights::getContextGradient(
      indices, contexts, context_vectors, weighted_representations,
      base_gradient);

  boost::shared_ptr<SourceFactoredWeights> gradient =
      dynamic_pointer_cast<SourceFactoredWeights>(base_gradient);
  assert(gradient != nullptr);

  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;
  int source_context_width = 2 * config->source_order - 1;

  MatrixReal context_gradients = MatrixReal::Zero(word_width, indices.size());
  for (int j = 0; j < source_context_width; ++j) {
    context_gradients =
        getSourceContextProduct(j, weighted_representations, true);

    int k = context_width + j;
    for (size_t i = 0; i < indices.size(); ++i) {
      gradient->SQ.col(contexts[i][k]) += context_gradients.col(i);
    }

    if (config->diagonal_contexts) {
      gradient->SC[j] += context_vectors[k].cwiseProduct(weighted_representations).rowwise().sum();
    } else {
      gradient->SC[j] += weighted_representations * context_vectors[k].transpose();
    }
  }
}

MatrixReal SourceFactoredWeights::getSourceContextProduct(
    int index, const MatrixReal& representations, bool transpose) const {
  if (config->diagonal_contexts) {
    return SC[index].asDiagonal() * representations;
  } else {
    if (transpose) {
      return SC[index].transpose() * representations;
    } else {
      return SC[index] * representations;
    }
  }
}

bool SourceFactoredWeights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<SourceFactoredWeights>& gradient,
    double eps) {
  if (!FactoredWeights::checkGradient(corpus, indices, gradient, eps)) {
    return false;
  }

  for (int i = 0; i < size; ++i) {
    SW(i) += eps;
    Real objective_plus = getObjective(corpus, indices);
    SW(i) -= eps;

    SW(i) -= eps;
    Real objective_minus = getObjective(corpus, indices);
    SW(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->SW(i) - est_gradient) > eps) {
      cout << i << " " << gradient->SW(i) << " " << est_gradient << endl;
      return false;
    }
  }

  return true;
}

void SourceFactoredWeights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<SourceFactoredWeights>& gradient) {
  FactoredWeights::syncUpdate(words, gradient);

  for (int word_id: words.getSourceWordsSet()) {
    lock_guard<mutex> lock(*mutexesSQ[word_id]);
    SQ.col(word_id) += gradient->SQ.col(word_id);
  }

  for (int i = 0; i < SC.size(); ++i) {
    lock_guard<mutex> lock(*mutexesSC[i]);
    SC[i] += gradient->SC[i];
  }
}

Block SourceFactoredWeights::getBlock(int start, int size) const {
  int thread_id = omp_get_thread_num();
  size_t block_size = size / config->threads + 1;
  size_t block_start = start + thread_id * block_size;
  block_size = min(block_size, start + size - block_start);

  return make_pair(block_start, block_size);
}

void SourceFactoredWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<SourceFactoredWeights>& global_gradient) {
  FactoredWeights::updateSquared(global_words, global_gradient);

  for (int word_id: global_words.getSourceWords()) {
    SQ.col(word_id).array() += global_gradient->SQ.col(word_id).array().square();
  }

  Block block = getBlock(SQ.size(), SW.size() - SQ.size());
  SW.segment(block.first, block.second).array() +=
      global_gradient->SW.segment(block.first, block.second).array().square();
}

void SourceFactoredWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<SourceFactoredWeights>& global_gradient,
    const boost::shared_ptr<SourceFactoredWeights>& adagrad) {
  FactoredWeights::updateAdaGrad(
      global_words, global_gradient, adagrad);

  for (int word_id: global_words.getSourceWords()) {
    SQ.col(word_id) -= global_gradient->SQ.col(word_id).binaryExpr(
        adagrad->SQ.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  Block block = getBlock(SQ.size(), SW.size() - SQ.size());
  SW.segment(block.first, block.second) -=
      global_gradient->SW.segment(block.first, block.second).binaryExpr(
          adagrad->SW.segment(block.first, block.second),
          CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real SourceFactoredWeights::regularizerUpdate(
    const boost::shared_ptr<SourceFactoredWeights>& global_gradient,
    Real minibatch_factor) {
  Real ret = FactoredWeights::regularizerUpdate(
      global_gradient, minibatch_factor);

  Block block = getBlock(0, SW.size());
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  SW.segment(block.first, block.second) -=
      SW.segment(block.first, block.second) * sigma;

  Real squares = SW.segment(block.first, block.second).array().square().sum();
  ret += 0.5 * minibatch_factor * config->l2_lbl * squares;

  return ret;
}

void SourceFactoredWeights::clear(
    const MinibatchWords& words, bool parallel_update) {
  FactoredWeights::clear(words, parallel_update);

  if (parallel_update) {
    for (int word_id: words.getSourceWords()) {
      SQ.col(word_id).setZero();
    }

    Block block = getBlock(SQ.size(), SW.size() - SQ.size());
    SW.segment(block.first, block.second).setZero();
  } else {
    for (int word_id: words.getSourceWordsSet()) {
      SQ.col(word_id).setZero();
    }

    SW.segment(SQ.size(), SW.size() - SQ.size()).setZero();
  }
}

SourceFactoredWeights::~SourceFactoredWeights() {
  delete data;
}

} // namespace oxlm
