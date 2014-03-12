#include "lbl/nlm_approximate_z.h"

#include "lbl/lbfgs2.h"

namespace oxlm {

Real NLMApproximateZ::z(const VectorReal& context) const {
  VectorReal z_products = context.transpose() * m_z_approx + m_b_approx.transpose(); // 1 x Z
  Real row_max = z_products.maxCoeff(); // 1 x 1
  VectorReal exp_z_products = (z_products.array() - row_max).exp(); // 1 x Z
  return log(exp_z_products.sum()) + row_max; // 1 x 1
}

void NLMApproximateZ::train(
    const MatrixReal& contexts, const VectorReal& zs,
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

} // namespace oxlm
