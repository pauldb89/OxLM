#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#include "log_bilinear_model.h"
#include "log_add.h"
#include "lbl/lbfgs2.h"

using namespace std;
using namespace boost;
using namespace oxlm;

static boost::mt19937 linear_model_rng(static_cast<unsigned> (std::time(0)));
static uniform_01<> linear_model_uniform_dist;

//LogBiLinearModel::LogBiLinearModel(const LogBiLinearModel& model) 
//  : config(model.config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), m_labels(model.label_set()) {
//    init(config, m_labels, false);
//    addModel(model);
//}

LogBiLinearModel::LogBiLinearModel(const ModelData& config, const Dict& labels, bool diagonal)
  : config(config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), M(0,0), m_labels(labels), m_diagonal(diagonal) {
    init(config, m_labels, true);
}

void LogBiLinearModel::init(const ModelData& config, const Dict& labels, bool init_weights) {
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
    std::cerr << " Created a LogBiLinearModel: "   << std::endl;
    std::cerr << "  Output Vocab size = "          << num_output_words << std::endl;
    std::cerr << "  Context Vocab size = "         << num_context_words << std::endl;
    std::cerr << "  Word Vector size = "           << word_width << std::endl;
    std::cerr << "  Context size = "               << context_width << std::endl;
    std::cerr << "  Diagonal = "                   << m_diagonal << std::endl;
    std::cerr << "  Total parameters = "           << m_data_size << std::endl;
    std::cerr << "===============================" << std::endl;
  }
}

void LogBiLinearModel::allocate_data(const ModelData& config) {
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


void LogBiLinearModelApproximateZ::train(const MatrixReal& contexts, const VectorReal& zs, 
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


void LogBiLinearModelApproximateZ::train_lbfgs(const MatrixReal& contexts, const VectorReal& zs, 
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
