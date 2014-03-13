#include "lbl/factored_nlm.h"

namespace oxlm {

FactoredNLM::FactoredNLM() {}

FactoredNLM::FactoredNLM(const ModelData& config, const Dict& labels)
    : NLM(config, labels, config.diagonal_contexts) {}

FactoredNLM::FactoredNLM(
    const ModelData& config, const Dict& labels,
    const vector<int>& classes)
    : NLM(config, labels, config.diagonal_contexts), indexes(classes),
      F(MatrixReal::Zero(config.classes, config.word_representation_size)),
      FB(VectorReal::Zero(config.classes)) {
  assert (!classes.empty());
  word_to_class.reserve(labels.size());
  for (size_t c = 0; c < classes.size() - 1; ++c) {
    int c_end = classes.at(c + 1);
    for (int w = classes.at(c); w < c_end; ++w) {
      word_to_class.push_back(c);
    }
  }
  assert (labels.size() == word_to_class.size());

  random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<Real> gaussian(0,0.1);
  for (int i = 0; i < F.rows(); i++) {
    FB(i) = gaussian(gen);
    for (int j = 0; j < F.cols(); j++)
      F(i, j) = gaussian(gen);
  }
}

Eigen::Block<WordVectorsType> FactoredNLM::class_R(const int c) {
  int c_start = indexes.at(c), c_end = indexes.at(c + 1);
  return R.block(c_start, 0, c_end - c_start, R.cols());
}

const Eigen::Block<const WordVectorsType> FactoredNLM::class_R(
    const int c) const {
  int c_start = indexes.at(c), c_end = indexes.at(c + 1);
  return R.block(c_start, 0, c_end - c_start, R.cols());
}

Eigen::VectorBlock<WeightsType> FactoredNLM::class_B(const int c) {
  int c_start = indexes.at(c), c_end = indexes.at(c + 1);
  return B.segment(c_start, c_end - c_start);
}

const Eigen::VectorBlock<const WeightsType> FactoredNLM::class_B(
    const int c) const {
  int c_start = indexes.at(c), c_end = indexes.at(c + 1);
  return B.segment(c_start, c_end - c_start);
}

int FactoredNLM::get_class(const WordId& w) const {
  assert(w >= 0 && w < int(word_to_class.size())
         && "ERROR: Failed to find word in class dictionary.");
  return word_to_class[w];
}

Real FactoredNLM::l2_gradient_update(Real sigma) {
  F -= F*sigma;
  FB -= FB*sigma;
  return NLM::l2_gradient_update(sigma) + F.array().square().sum() + FB.array().square().sum();
}

void FactoredNLM::reclass(vector<WordId>& train, vector<WordId>& test) {
  cerr << "\n Reallocating classes:" << endl;
  MatrixReal class_dot_products = R * F.transpose();
  VectorReal magnitudes = F.rowwise().norm().transpose();
  vector< vector<int> > new_classes(F.rows());

  cerr << magnitudes << endl;
  cerr << magnitudes.rows() << " " << magnitudes.cols() << endl;
  cerr << class_dot_products.rows() << " " << class_dot_products.cols() << endl;


  for (int w_id=0; w_id < R.rows(); ++w_id) {
    int new_class=0;
    (class_dot_products.row(w_id).array() / magnitudes.transpose().array()).maxCoeff(&new_class);
    new_classes.at(new_class).push_back(w_id);
  }

  Dict new_dict;
  vector<int> new_word_to_class(word_to_class.size());
  vector<int> new_indexes(indexes.size());
  new_indexes.at(0) = 0;
  new_indexes.at(1) = 2;

  MatrixReal old_R=R, old_Q=Q;
  int moved_words=0;
  for (int c_id=0; c_id < int(new_classes.size()); ++c_id) {
    cerr << "  " << c_id << ": " << new_classes.at(c_id).size() << " word types, ending at ";
    for (auto w_id : new_classes.at(c_id)) {
      // re-index the word in the new dict
      string w_s = label_str(w_id);
      WordId new_w_id = new_dict.Convert(w_s);

      // reposition the word's representation vectors
      R.row(new_w_id) = old_R.row(w_id);
      Q.row(new_w_id) = old_Q.row(w_id);

      if (w_s != "</s>") {
        new_indexes.at(c_id+1) = new_w_id+1;
        new_word_to_class.at(new_w_id) = c_id;
      }
      else
        new_word_to_class.at(new_w_id) = 0;

      if (get_class(w_id) != c_id)
        ++moved_words;
    }
    cerr << new_indexes.at(c_id+1) << endl;
  }
  cerr << "\n " << moved_words << " word types moved class." << endl;

  swap(word_to_class, new_word_to_class);
  swap(indexes, new_indexes);

  for (WordId& w_id : train)
    w_id = new_dict.Lookup(label_str(w_id));
  for (WordId& w_id : test)
    w_id = new_dict.Lookup(label_str(w_id));

  m_labels = new_dict;
}

Real FactoredNLM::log_prob(
    const WordId w, const vector<WordId>& context,
    bool non_linear, bool cache) const {
  VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
  int width = config.ngram_order-1;
  int gap = width-context.size();
  assert(static_cast<int>(context.size()) <= width);
  for (int i=gap; i < width; i++) {
    if (m_diagonal) {
      prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
    } else {
      prediction_vector += Q.row(context.at(i-gap)) * C.at(i);
    }
  }

  int c = get_class(w);
  // a simple non-linearity
  if (non_linear)
    prediction_vector = sigmoid(prediction_vector);

  // log p(c | context)
  Real class_log_prob = 0;
  pair<unordered_map<Words, Real, container_hash<Words> >::iterator, bool> context_cache_result;
  if (cache) {
    context_cache_result = m_context_cache.insert(make_pair(context,0));
  }
  if (cache && !context_cache_result.second) {
    assert (context_cache_result.first->second != 0);
    class_log_prob = F.row(c)*prediction_vector + FB(c) - context_cache_result.first->second;
  } else {
    Real c_log_z=0;
    VectorReal class_probs = logSoftMax(F*prediction_vector + FB, &c_log_z);
    assert(c_log_z != 0);
    class_log_prob = class_probs(c);
    if (cache) {
      context_cache_result.first->second = c_log_z;
    }
  }

  // log p(w | c, context)
  Real word_log_prob = 0;
  pair<unordered_map<pair<int,Words>, Real>::iterator, bool> class_context_cache_result;
  if (cache) {
    class_context_cache_result = m_context_class_cache.insert(make_pair(make_pair(c,context),0));
  }
  if (cache && !class_context_cache_result.second) {
    word_log_prob = R.row(w)*prediction_vector + B(w) - class_context_cache_result.first->second;
  } else {
    int c_start = indexes.at(c);
    Real w_log_z=0;
    VectorReal word_probs = logSoftMax(class_R(c)*prediction_vector + class_B(c), &w_log_z);
    word_log_prob = word_probs(w-c_start);
    if (cache) {
      class_context_cache_result.first->second = w_log_z;
    }
  }

  return class_log_prob + word_log_prob;
}

void FactoredNLM::clear_cache() {
  m_context_cache.clear();
  m_context_cache.reserve(1000000);
  m_context_class_cache.clear();
  m_context_class_cache.reserve(1000000);
}

}
