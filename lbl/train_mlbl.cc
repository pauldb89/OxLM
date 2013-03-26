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

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "lbl/lbfgs.h"
#include "lbl/log_bilinear_model.h"
#include "lbl/log_add.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;


typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

void learn(const variables_map& vm, const ModelData& config);

Real function_and_gradient(LogBiLinearModel& model,
                           int start, int end, const Corpus& training_corpus,
                           Real lambda,
                           LogBiLinearModel::WeightsType& gradient,
                           Real* gradient_data, 
                           Real& wnorm, Real& gnorm);

Real perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride=1);


int main(int argc, char **argv) {
  cout << "LBFGS optimisation for a mixture of log-bilinear models: Copyright 2013 Phil Blunsom, " 
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
    ("input,i", value<string>()->default_value("data.txt"), 
        "corpus of sentences, one per line")
    ("test-set", value<string>(), 
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10), 
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(100), 
        "number of sentences per minibatch")
    ("order,n", value<int>()->default_value(3), 
        "ngram order")
    ("model-in,m", value<string>(), 
        "initial model")
    ("model-out,o", value<string>()->default_value("model"), 
        "base filename of model output files")
    ("lambda,r", value<float>()->default_value(0.0), 
        "regularisation strength parameter")
    ("dump-frequency", value<int>()->default_value(0), 
        "dump model every n minibatches.")
    ("label-sample-size,s", value<int>()->default_value(100), 
        "number of previous labels to cache for sampling the partition function.")
    ("word-width", value<int>()->default_value(100), 
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("test-tokens", value<int>()->default_value(10000), 
        "number of evenly space test points tokens evaluate.")
    ("step-size", value<float>()->default_value(1.0), 
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
    ("randomise", "visit the training tokens in random order")
    ("uniform", "sample noise distribution from a uniform (default unigram) distribution.")
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

  ModelData config;
  config.label_sample_size = vm["label-sample-size"].as<int>();
  config.l2_parameter = vm["lambda"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  config.verbose = vm.count("verbose");
  config.uniform = vm.count("uniform");

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  cerr << "# order = " << vm["order"].as<int>() << endl;
  if (vm.count("model-in"))
    cerr << "# model-in = " << vm["model-in"].as<string>() << endl;
  cerr << "# model-out = " << vm["model-out"].as<string>() << endl;
  cerr << "# input = " << vm["input"].as<string>() << endl;
  cerr << "# minibatch-size = " << vm["minibatch-size"].as<int>() << endl;
  cerr << "# lambda = " << vm["lambda"].as<float>() << endl;
  cerr << "# label-sample-size = " << vm["label-sample-size"].as<int>() << endl;
  cerr << "# iterations = " << vm["iterations"].as<int>() << endl;
  cerr << "# threads = " << vm["threads"].as<int>() << endl;
  cerr << "################################" << endl;

  omp_set_num_threads(config.threads);

  learn(vm, config);

  return 0;
}


void learn(const variables_map& vm, const ModelData& config) {
  Corpus training_corpus, test_corpus;

  //////////////////////////////////////////////
  // read the training sentences
  ifstream in(vm["input"].as<string>().c_str());
  string line, token;

  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");

  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token) 
      training_corpus.push_back(dict.Convert(token));
    training_corpus.push_back(end_id);
  }
  in.close();
  //////////////////////////////////////////////
  
  //////////////////////////////////////////////
  // read the test sentences
  bool have_test = vm.count("test-set");
  if (have_test) {
    ifstream test_in(vm["test-set"].as<string>().c_str());
    while (getline(test_in, line)) {
      stringstream line_stream(line);
      Sentence tokens;
      while (line_stream >> token)
        test_corpus.push_back(dict.Convert(token, true));
      test_corpus.push_back(end_id);
    }
    test_in.close();
  }
  //////////////////////////////////////////////

  LogBiLinearModel model(config, dict);

  if (vm.count("model-in")) {
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;
  }

  int num_weights = model.num_weights();
  Real* gradient_data = new Real[num_weights];
  LogBiLinearModel::WeightsType gradient(gradient_data, num_weights);

//  int thread_id = omp_get_thread_num();
  Real lambda = config.l2_parameter; 

  Real f=0.0, wnorm=0.0, gnorm=numeric_limits<Real>::max();

  scitbx::lbfgs::minimizer<Real> minimiser(num_weights, 10);
  int function_evaluations=0;
  bool calc_g_and_f = true;
  int lbfgs_iteration = minimiser.iter();
  clock_t lbfgs_time=0, gradient_time=0;
  while (lbfgs_iteration < vm["lbfgs-iterations"].as<int>() && gnorm > vm["gnorm-threshold"].as<float>()) {
    if (calc_g_and_f) {
      clock_t gradient_start = clock();

      gradient.setZero();
      f = function_and_gradient(model, 0, training_corpus.size(), training_corpus, lambda, gradient, gradient_data, wnorm, gnorm);
      function_evaluations++;
      gradient_time += (clock() - gradient_start);
    }

    if (lbfgs_iteration == 0 || (!calc_g_and_f )) {
      cerr << "  (" << lbfgs_iteration+1 << "." << function_evaluations << ":" 
        << "f=" << f << ",|w|=" << wnorm << ",|g|=" << gnorm << ")\n";
    }

    clock_t lbfgs_start=clock();
    calc_g_and_f = minimiser.run(model.data(), f, gradient_data);
    lbfgs_iteration = minimiser.iter();
    lbfgs_time += (clock() - lbfgs_start);
  }

  minimiser.run(model.data(), f, gradient_data);
  if (vm.count("test-set"))
    cerr << ", Test Perplexity = " << perplexity(model, test_corpus, test_corpus.size()/vm["test-tokens"].as<int>()); 
  cerr << " |" << endl << endl;

  if (vm.count("model-out")) {
    cout << "Writing trained model to " << vm["model-out"].as<string>() << endl;
    std::ofstream f(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f);
    ar << model;
  }

  delete gradient_data;
}


Real function_and_gradient(LogBiLinearModel& model,
                           int start, int end, const Corpus& training_corpus,
                           Real lambda,
                           LogBiLinearModel::WeightsType& gradient,
                           Real* gradient_data, 
                           Real& wnorm, Real& gnorm) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;
  int num_words = model.labels();

  int R_size = num_words*word_width;
  int Q_size = R_size;
  int C_size = word_width*word_width;
  int B_size = num_words;

  assert((R_size+Q_size+context_width*C_size+B_size) == model.num_weights());

  LogBiLinearModel::WordVectorsType g_R(gradient_data, num_words, word_width);
  LogBiLinearModel::WordVectorsType g_Q(gradient_data+R_size, num_words, word_width);

  LogBiLinearModel::ContextTransformsType g_C;
  Real* ptr = gradient_data+2*R_size;
  for (int i=0; i<context_width; i++) {
    g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, word_width));
    ptr += C_size;
  }

  LogBiLinearModel::WeightsType g_B(ptr, num_words);

  // cache the products of Q with the contexts 
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.Q * model.C.at(i);

  // cache the partition functions
  std::vector<VectorReal> log_partition_functions(context_width, VectorReal(num_words));
  std::vector<MatrixReal> expected_representations(context_width, MatrixReal(num_words, context_width));
  for (int i=0; i<context_width; i++) { // O(context_width * |v|^2)
    for (int q=0; q<model.labels(); ++q) { // O(|v|^2)
      VectorReal logProbs = model.R * q_context_products[i].row(q).transpose();// + model.B;
      Real max_logProb = logProbs.maxCoeff();
      Real logProbs_z = log((logProbs.array() - max_logProb).exp().sum()) + max_logProb;

      assert(isfinite(logProbs_z));
      logProbs.array() -= logProbs_z;

      log_partition_functions[i](q) = logProbs_z;

      // model update component
      VectorReal probs = logProbs.array().exp();

      VectorReal expected_features = (probs * q_context_products[i].row(q)); // O(|v|)
      g_R += expected_features;
      expected_representations[i].row(q) = expected_features.colwise().sum();
    }
  }


  return f;
}


Real perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride) {
  Real p=0.0;
  Real log_z_sum=0.0;

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // cache the products of Q with the contexts 
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.Q * model.C.at(i);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
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
      for (int i=0; i<context_width; ++i) {
        int j=context_start+i;
        int v_i = (j<0 ? start_id : test_corpus.at(j));
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
    }
  }
  cerr << ", Average log_z = " << log_z_sum / tokens;
  return exp(-p/tokens);
}
