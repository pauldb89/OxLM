#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#include "nlm.h"
#include "log_add.h"
#include "lbl/lbfgs2.h"
#include "lbl/feature_generator.h"

using namespace std;
using namespace boost;
using namespace oxlm;

static boost::mt19937 linear_model_rng(static_cast<unsigned> (std::time(0)));
static uniform_01<> linear_model_uniform_dist;

//NLM::NLM(const NLM& model) 
//  : config(model.config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), m_labels(model.label_set()) {
//    init(config, m_labels, false);
//    addModel(model);
//}

NLM::NLM(const ModelData& config, const Dict& labels, bool diagonal)
  : config(config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), M(0,0), m_labels(labels), m_diagonal(diagonal) {
    init(config, m_labels, true);
  }

void NLM::init(const ModelData& config, const Dict& labels, bool init_weights) {
  // the prediction vector ranges over classes for a class based LM, or the vocab otherwise
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int C_size = (m_diagonal ? word_width : word_width*word_width);
  //int C_size;
  //if (m_diagonal) C_size = word_width;
  //else            C_size = word_width*word_width;

  allocate_data(config);

  new (&W) WeightsType(m_data, m_data_size);
  if (init_weights) {
    //    W.setRandom() /= 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i=0; i<m_data_size; i++)
      W(i) = gaussian(gen);
  }
  else W.setZero();

  new (&R) WordVectorsType(m_data, num_output_words, word_width);
  new (&Q) WordVectorsType(m_data+R_size, num_context_words, word_width);

  C.clear();
  Real* ptr = m_data+R_size+Q_size;
  for (int i=0; i<context_width; i++) {
    if (m_diagonal) C.push_back(ContextTransformType(ptr, word_width, 1));
    else            C.push_back(ContextTransformType(ptr, word_width, word_width));
    ptr += C_size;
    //     C.back().setIdentity();
    //      C.back().setZero();
  }

  new (&B) WeightsType(ptr, num_output_words);
  new (&M) WeightsType(ptr+num_output_words, context_width);

  M.setZero();

  //R.setOnes();
  //R.setZero();
  //R.setIdentity();
  //Q.setOnes();
  //Q.setIdentity();
  //Q.setZero();
  //B.setOnes();
  //    R << 0,0,0,1 , 0,0,1,0 , 0,1,0,0 , 1,0,0,0; 
  //    R << 0,0 , 0,0 , 0,1 , 1,0; 
  //    Q << 0,0,0,1 , 0,0,1,0 , 0,1,0,0 , 1,0,0,0; 
  //    Q << 1,1 , 1,1 , 1,1 , 1,1; 

  //  assert(ptr+num_output_words == m_data+m_data_size); 

#pragma omp master
  if (true) {
    std::cerr << "===============================" << std::endl;
    std::cerr << " Created a NLM: "   << std::endl;
    std::cerr << "  Output Vocab size = "          << num_output_words << std::endl;
    std::cerr << "  Context Vocab size = "         << num_context_words << std::endl;
    std::cerr << "  Word Vector size = "           << word_width << std::endl;
    std::cerr << "  Context size = "               << context_width << std::endl;
    std::cerr << "  Diagonal = "                   << m_diagonal << std::endl;
    std::cerr << "  Total parameters = "           << m_data_size << std::endl;
    std::cerr << "===============================" << std::endl;
  }
}

void NLM::allocate_data(const ModelData& config) {
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int C_size = (m_diagonal ? word_width : word_width*word_width);
  int B_size = num_output_words;
  int M_size = context_width;

  m_data_size = R_size + Q_size + context_width*C_size + B_size + M_size;
  m_data = new Real[m_data_size];
}


void NLMApproximateZ::train(const MatrixReal& contexts, const VectorReal& zs, 
                            Real step_size, int iterations, int approx_vectors) {
  int word_width = contexts.cols();
  m_z_approx = MatrixReal(word_width, approx_vectors); // W x Z
  m_b_approx = VectorReal(approx_vectors); // Z x 1
  { // z_approx initialisation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int j=0; j<m_z_approx.cols(); j++) {
      m_b_approx(j) = gaussian(gen);
      for (int i=0; i<m_z_approx.rows(); i++)
        m_z_approx(i,j) = gaussian(gen);
    }
  }
  //  z_approx.col(0) = sol;
  MatrixReal train_ps = contexts;
  VectorReal train_zs = zs;

  MatrixReal z_adaGrad = MatrixReal::Zero(m_z_approx.rows(), m_z_approx.cols());
  VectorReal b_adaGrad = VectorReal::Zero(m_b_approx.rows());
  for (int iteration=0; iteration < iterations; ++iteration) {
    MatrixReal z_products = (train_ps * m_z_approx).rowwise() + m_b_approx.transpose(); // n x Z
    VectorReal row_max = z_products.rowwise().maxCoeff(); // n x 1
    MatrixReal exp_z_products = (z_products.colwise() - row_max).array().exp(); // n x Z
    VectorReal pred_zs = (exp_z_products.rowwise().sum()).array().log() + row_max.array(); // n x 1

    VectorReal err_gr = 2.0 * (train_zs - pred_zs); // n x 1
    MatrixReal probs = (z_products.colwise() - pred_zs).array().exp(); //  n x Z

    MatrixReal z_gradient = (-train_ps).transpose() * err_gr.asDiagonal() * probs; // W x Z
    z_adaGrad.array() += z_gradient.array().square();
    m_z_approx.array() -= step_size*z_gradient.array()/z_adaGrad.array().sqrt();

    VectorReal b_gradient = err_gr.transpose() * probs; // Z x 1
    b_adaGrad.array() += b_gradient.array().square();
    m_b_approx.array() -= step_size*b_gradient.array()/b_adaGrad.array().sqrt();

    if (iteration % 10 == 0) {
      cerr << iteration << " : Train NLLS = " << (train_zs - pred_zs).squaredNorm() / train_zs.rows();
      //      Real diff = train_zs.sum() - pred_zs.sum();
      //      Real new_pp = exp(-(train_pp + train_zs.sum() - pred_zs.sum())/train_corpus.size());
      //      cerr << ", PPL = " << new_pp << ", z_diff = " << diff;
      cerr << endl;
      /*
         MatrixReal test_z_products = (test_ps * z_approx).rowwise() + b_approx.transpose(); // n x Z
         VectorReal test_row_max = test_z_products.rowwise().maxCoeff(); // n x 1
         MatrixReal test_exp_z_products = (test_z_products.colwise() - test_row_max).array().exp(); // n x Z
         VectorReal test_pred_zs = (test_exp_z_products.rowwise().sum()).array().log() + test_row_max.array(); // n x 1

         cerr << ", Test NLLS = " << (test_zs - test_pred_zs).squaredNorm() / test_zs.rows();
         diff = test_zs.sum() - test_pred_zs.sum();
         new_pp = exp(-(test_pp + test_zs.sum() - test_pred_zs.sum())/test_corpus.size());
         cerr << ", Test PPL = " << new_pp << ", z_diff = " << diff << endl;
         */
    }
  }
}


void NLMApproximateZ::train_lbfgs(const MatrixReal& contexts, const VectorReal& zs, 
                                  Real step_size, int iterations, int approx_vectors) {
  int word_width = contexts.cols();
  m_z_approx = MatrixReal(word_width, approx_vectors); // W x Z
  m_b_approx = VectorReal(approx_vectors); // Z x 1
  { // z_approx initialisation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int j=0; j<m_z_approx.cols(); j++) {
      m_b_approx(j) = gaussian(gen);
      for (int i=0; i<m_z_approx.rows(); i++)
        m_z_approx(i,j) = gaussian(gen);
    }
  }

  MatrixReal train_ps = contexts;
  VectorReal train_zs = zs;

  int z_weights = m_z_approx.rows()*m_z_approx.cols();
  int b_weights = m_b_approx.rows();
  Real *weights_data = new Real[z_weights+b_weights]; 
  Real *gradient_data = new Real[z_weights+b_weights]; 
  memcpy(weights_data, m_z_approx.data(), sizeof(Real)*z_weights);
  memcpy(weights_data+z_weights, m_b_approx.data(), sizeof(Real)*b_weights);

  scitbx::lbfgs::minimizer<Real>* minimiser = new scitbx::lbfgs::minimizer<Real>(z_weights+b_weights, 50);

  bool calc_g_and_f=true;
  Real f=0;
  int function_evaluations=0;
  for (int iteration=0; iteration < iterations;) {
    if (calc_g_and_f) {
      MatrixReal z_products = (train_ps * m_z_approx).rowwise() + m_b_approx.transpose(); // n x Z
      VectorReal row_max = z_products.rowwise().maxCoeff(); // n x 1
      MatrixReal exp_z_products = (z_products.colwise() - row_max).array().exp(); // n x Z
      VectorReal pred_zs = (exp_z_products.rowwise().sum()).array().log() + row_max.array(); // n x 1

      VectorReal err_gr = 2.0 * (train_zs - pred_zs); // n x 1
      MatrixReal probs = (z_products.colwise() - pred_zs).array().exp(); //  n x Z

      MatrixReal z_gradient = (-train_ps).transpose() * err_gr.asDiagonal() * probs; // W x Z
      VectorReal b_gradient = err_gr.transpose() * probs; // Z x 1
      memcpy(gradient_data, z_gradient.data(), sizeof(Real)*z_weights);
      memcpy(gradient_data+z_weights, b_gradient.data(), sizeof(Real)*b_weights);

      f = (train_zs - pred_zs).squaredNorm();

      function_evaluations++;
    }

    //if (iteration == 0 || (!calc_g_and_f ))
    cerr << "  (" << iteration+1 << "." << function_evaluations << ":" << "f=" << f / train_zs.rows() << ")\n";

    try { 
      calc_g_and_f = minimiser->run(weights_data, f, gradient_data); 
      memcpy(m_z_approx.data(), weights_data, sizeof(Real)*z_weights);
      memcpy(m_b_approx.data(), weights_data+z_weights, sizeof(Real)*b_weights);
    }
    catch (const scitbx::lbfgs::error &e) {
      cerr << "LBFGS terminated with error:\n  " << e.what() << "\nRestarting..." << endl;
      delete minimiser;
      minimiser = new scitbx::lbfgs::minimizer<Real>(z_weights+b_weights, 50);
      calc_g_and_f = true;
    }
    iteration = minimiser->iter();
  }

  minimiser->run(weights_data, f, gradient_data);
  memcpy(m_z_approx.data(), weights_data, sizeof(Real)*z_weights);
  memcpy(m_b_approx.data(), weights_data+z_weights, sizeof(Real)*b_weights);
  delete minimiser;
  delete weights_data;
  delete gradient_data;
}


FactoredOutputNLM::FactoredOutputNLM(const ModelData& config, 
                                     const Dict& labels, 
                                     bool diagonal, 
                                     const std::vector<int>& classes) 
: NLM(config, labels, diagonal), indexes(classes), 
  F(MatrixReal::Zero(config.classes, config.word_representation_size)),
  FB(VectorReal::Zero(config.classes)) {
    assert (!classes.empty());
    word_to_class.reserve(labels.size());
    for (int c=0; c < int(classes.size())-1; ++c) {
      int c_end = classes.at(c+1);
      //cerr << "\nClass " << c << ":" << endl;
      for (int w=classes.at(c); w < c_end; ++w) {
        word_to_class.push_back(c);
        //cerr << " " << label_str(w);
      }
    }
    assert (labels.size() == word_to_class.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i=0; i<F.rows(); i++) {
      FB(i) = gaussian(gen);
      for (int j=0; j<F.cols(); j++)
        F(i,j) = gaussian(gen);
    }
  }

void FactoredOutputNLM::reclass(vector<WordId>& train, vector<WordId>& test) {
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
  std::vector<int> new_word_to_class(word_to_class.size());
  std::vector<int> new_indexes(indexes.size());
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

FactoredMaxentNLM::FactoredMaxentNLM(
    const ModelData& config, const Dict& labels, bool diagonal) :
    FactoredOutputNLM(config, labels, diagonal) {}

FactoredMaxentNLM::FactoredMaxentNLM(
    const ModelData& config, const Dict& labels,
    bool diagonal, const std::vector<int>& classes) :
    FactoredOutputNLM(config, labels, diagonal, classes),
    U(config.classes) {
  for (int i = 0; i < config.classes; ++i) {
    V.push_back(UnconstrainedFeatureStore(classes[i + 1] - classes[i]));
  }
}


Real FactoredMaxentNLM::log_prob(
    const WordId w, const std::vector<WordId>& context,
    bool non_linear=false, bool cache=false) const {
  VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
  int width = config.ngram_order-1;
  int gap = width-context.size();
  assert(gap == 0);
  assert(static_cast<int>(context.size()) <= width);
  for (int i=gap; i < width; i++)
    if (m_diagonal) prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
    else            prediction_vector += Q.row(context.at(i-gap)) * C.at(i);

  int c = get_class(w);
  int c_start = indexes.at(c);
  FeatureGenerator generator(config.feature_context_size);
  vector<Feature> features = generator.generate(context);
  VectorReal class_feature_scores = U.get(features);
  VectorReal word_feature_scores = V[c].get(features);

  // a simple non-linearity
  if (non_linear)
    prediction_vector = sigmoid(prediction_vector);

  // log p(c | context) 
  Real class_log_prob = 0;
  std::pair<std::unordered_map<Words, Real, container_hash<Words> >::iterator, bool> context_cache_result;
  if (cache) context_cache_result = m_context_cache.insert(make_pair(context,0));
  if (cache && !context_cache_result.second) {
    assert (context_cache_result.first->second != 0);
    class_log_prob = F.row(c)*prediction_vector + FB(c) + class_feature_scores(c) - context_cache_result.first->second;
  } else {
    Real c_log_z=0;
    VectorReal class_probs = logSoftMax(F*prediction_vector + FB + class_feature_scores, &c_log_z);
    assert(c_log_z != 0);
    class_log_prob = class_probs(c);
    if (cache) {
      context_cache_result.first->second = c_log_z;
    }
  }

  // log p(w | c, context) 
  Real word_log_prob = 0;
  std::pair<std::unordered_map<std::pair<int,Words>, Real>::iterator, bool> class_context_cache_result;
  if (cache) class_context_cache_result = m_context_class_cache.insert(make_pair(make_pair(c,context),0));

  if (cache && !class_context_cache_result.second) {
    word_log_prob  = R.row(w)*prediction_vector + B(w) + word_feature_scores(w - c_start) - class_context_cache_result.first->second;
  } else {
    Real w_log_z = 0;
    VectorReal word_probs = logSoftMax(class_R(c) * prediction_vector + class_B(c) + word_feature_scores, &w_log_z);
    word_log_prob = word_probs(w - c_start);
    if (cache) {
      class_context_cache_result.first->second = w_log_z;
    }
  }

  return class_log_prob + word_log_prob;
}
