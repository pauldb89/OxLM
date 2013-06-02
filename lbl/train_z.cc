// STL
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iterator>
#include <cstring>
#include <functional>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <set>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "lbl/log_bilinear_model.h"
#include "lbl/log_add.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;


Real extract_p(const LogBiLinearModel& model, const Corpus& test_corpus, MatrixReal& ps, VectorReal& zs, int stride=1);


int main(int argc, char **argv) {
  cout << "Appoximating the partition function of a log-bilinear models: Copyright 2013 Phil Blunsom, " 
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm; 

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(), 
        "config file specifying additional command line options")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("test-set", value<string>(), 
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10), 
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(100), 
        "number of sentences per minibatch")
    ("model-in,m", value<string>(), 
        "initial model")
    ("model-out,m", value<string>(), 
        "file to serialise the approximation model to")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("approx-vectors", value<int>()->default_value(1), 
        "number of approximation terms (n) in \\sum_i b_i^n exp(p.z_i).")
    ("step-size", value<float>()->default_value(1.0), 
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
    ("randomise", "visit the training tokens in random order")
    ("uniform", "sample noise distribution from a uniform (default unigram) distribution.")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ("mixture", "Train a mixture of bigram LBL models, one per context position.")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm); 
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm); 
  }
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help")) { 
    cout << cmdline_options << "\n"; 
    return 1; 
  }

  omp_set_num_threads(vm["threads"].as<int>());

  Dict dict;
  ModelData config;
  LogBiLinearModel model(config, dict);

  assert(vm.count("model-in"));
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;
  f.close();

  dict = model.label_set();
  
  //////////////////////////////////////////////
  // read the test sentences
  Corpus corpus;
  string line, token;
  WordId end_id = dict.Convert("</s>");
  assert (vm.count("test-set"));
  ifstream test_in(vm["test-set"].as<string>().c_str());
  while (getline(test_in, line)) {
    stringstream line_stream(line);
    Sentence tokens;
    while (line_stream >> token) {
      WordId w = dict.Convert(token, true);
      if (w < 0) {
        cerr << token << " " << w << endl;
        assert(!"Unknown word found in test corpus.");
      }
      corpus.push_back(w);
    }
    corpus.push_back(end_id);
  }
  test_in.close();
  //////////////////////////////////////////////
  Corpus train_corpus(corpus.begin(), corpus.begin()+(int)(corpus.size()*0.8));
  Corpus test_corpus(corpus.begin()+train_corpus.size(), corpus.end());
  
  int word_width = model.config.word_representation_size;

  VectorReal train_zs(train_corpus.size());
  MatrixReal train_ps(train_corpus.size(), word_width);

  VectorReal test_zs(test_corpus.size());
  MatrixReal test_ps(test_corpus.size(), word_width);

  Real train_pp = extract_p(model, train_corpus, train_ps, train_zs);
  Real test_pp = extract_p(model, test_corpus, test_ps, test_zs);

  cerr << "  Train Perplexity = " << exp(-train_pp/train_corpus.size()) << "(" << train_zs.rows() << " tokens )" << endl; 
  cerr << "  Test Perplexity = " << exp(-test_pp/test_corpus.size()) << "(" << test_zs.rows() << " tokens )" << endl; 

  Real log_z_av=train_zs.mean();
  Real log_z_var = (train_zs.array() - log_z_av).matrix().squaredNorm() / train_zs.rows();
  cerr << "  Train average log_z = " << log_z_av << ", Variance = " << log_z_var << endl;

  VectorReal sol = train_ps.jacobiSvd(ComputeThinU | ComputeThinV).solve(train_zs);

  Real log_ls_var = ((train_ps * sol) - train_zs).squaredNorm() / train_zs.rows();
  cerr << "  Train LeastSquares Variance = " << log_ls_var << endl;
/*
  MatrixReal z_approx(word_width, vm["approx-vectors"].as<int>()); // W x Z
  VectorReal b_approx(vm["approx-vectors"].as<int>()); // Z x 1
  { // z_approx initialisation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int j=0; j<z_approx.cols(); j++) {
      b_approx(j) = gaussian(gen);
      for (int i=0; i<z_approx.rows(); i++)
        z_approx(i,j) = gaussian(gen);
    }
  }
  z_approx.col(0) = sol;

  MatrixReal z_adaGrad = MatrixReal::Zero(z_approx.rows(), z_approx.cols());
  VectorReal b_adaGrad = VectorReal::Zero(b_approx.rows());
  Real step_size = vm["step-size"].as<float>();
  for (int iteration=0; iteration<vm["iterations"].as<int>(); ++iteration) {
    MatrixReal z_products = (train_ps * z_approx).rowwise() + b_approx.transpose(); // n x Z
    VectorReal row_max = z_products.rowwise().maxCoeff(); // n x 1
    MatrixReal exp_z_products = (z_products.colwise() - row_max).array().exp(); // n x Z
    VectorReal pred_zs = (exp_z_products.rowwise().sum()).array().log() + row_max.array(); // n x 1

    VectorReal err_gr = 2.0 * (train_zs - pred_zs); // n x 1
    MatrixReal probs = (z_products.colwise() - pred_zs).array().exp(); //  n x Z

    MatrixReal z_gradient = (-train_ps).transpose() * err_gr.asDiagonal() * probs; // W x Z
    z_adaGrad.array() += z_gradient.array().square();
    z_approx.array() -= step_size*z_gradient.array()/z_adaGrad.array().sqrt();

    VectorReal b_gradient = err_gr.transpose() * probs; // Z x 1
    b_adaGrad.array() += b_gradient.array().square();
    b_approx.array() -= step_size*b_gradient.array()/b_adaGrad.array().sqrt();

    if (iteration % 100 == 0) {
      cerr << iteration << " : Train NLLS = " << (train_zs - pred_zs).squaredNorm() / train_zs.rows();
      Real diff = train_zs.sum() - pred_zs.sum();
      Real new_pp = exp(-(train_pp + train_zs.sum() - pred_zs.sum())/train_corpus.size());
      cerr << ", PPL = " << new_pp << ", z_diff = " << diff;

      MatrixReal test_z_products = (test_ps * z_approx).rowwise() + b_approx.transpose(); // n x Z
      VectorReal test_row_max = test_z_products.rowwise().maxCoeff(); // n x 1
      MatrixReal test_exp_z_products = (test_z_products.colwise() - test_row_max).array().exp(); // n x Z
      VectorReal test_pred_zs = (test_exp_z_products.rowwise().sum()).array().log() + test_row_max.array(); // n x 1

      cerr << ", Test NLLS = " << (test_zs - test_pred_zs).squaredNorm() / test_zs.rows();
      diff = test_zs.sum() - test_pred_zs.sum();
      new_pp = exp(-(test_pp + test_zs.sum() - test_pred_zs.sum())/test_corpus.size());
      cerr << ", Test PPL = " << new_pp << ", z_diff = " << diff << endl;
    }
  }
*/
  LogBiLinearModelApproximateZ approximation;
  approximation.train(train_ps, train_zs, vm["step-size"].as<float>(), 
                      vm["iterations"].as<int>(),vm["approx-vectors"].as<int>());

  if (vm.count("model-out")) {
    std::ofstream f_out(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f_out);
    cerr << " Serialising z approximation ..."; cerr.flush();
    ar << approximation;
    cerr << " done." << endl;
    f_out.close();
  }

  return 0;
}


Real extract_p(const LogBiLinearModel& model, const Corpus& test_corpus, MatrixReal& ps, VectorReal& zs, int stride) {
  Real p=0.0;
  Real log_z_sum=0.0;

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  zs = VectorReal(test_corpus.size());
  ps = MatrixReal(test_corpus.size(), word_width);

  // cache the products of Q with the contexts 
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.context_product(i, model.Q);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");
  #pragma omp parallel \
      shared(test_corpus,model,stride,q_context_products,word_width) \
      reduction(+:p,log_z_sum,tokens) 
  {
    VectorReal prediction_vector(word_width);
    size_t thread_num = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    for (size_t s = (thread_num*stride); s < test_corpus.size(); s += (num_threads*stride)) {
      WordId w = test_corpus.at(s);
      prediction_vector.setZero();

      int context_start = s - context_width;
      bool sentence_start = (s==0);
      for (int i=context_width-1; i>=0; --i) {
        int j=context_start+i;
        sentence_start = (sentence_start || j<0 || test_corpus.at(j) == end_id);
        int v_i = (sentence_start ? start_id : test_corpus.at(j));
        prediction_vector += q_context_products[i].row(v_i).transpose();
      }

      ArrayReal score_vector = model.R * prediction_vector + model.B;
      Real w_p = score_vector(w);
      Real max_score = score_vector.maxCoeff();
      Real log_z = log((score_vector-max_score).exp().sum()) + max_score;
      w_p -= log_z;
      log_z_sum += log_z;
      p += w_p;

      tokens++;

      zs(s) = log_z;
      ps.row(s) = prediction_vector;
    }
  }
  return p;
}
