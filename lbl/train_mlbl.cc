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
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/progress.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "lbl/lbfgs.h"
#include "lbl/lbfgs2.h"
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

Real function_and_gradient(LogBiLinearModel& model, const Corpus& training_corpus,
                           Real lambda, LogBiLinearModel::WeightsType& gradient, 
                           Real* gradient_data, Real& wnorm, Real& gnorm);

Real perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride=1);


inline VectorReal softMax(const VectorReal& v) {
  Real max = v.maxCoeff();
  return (v.array() - (log((v.array() - max).exp().sum()) + max)).exp();
}


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
    ("input,i", value<string>(), 
        "corpus of sentences, one per line")
    ("test-set", value<string>(), 
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10), 
        "number of passes through the data")
    ("order,n", value<int>()->default_value(3), 
        "ngram order")
    ("model-in,m", value<string>(), 
        "initial model")
    ("model-out,o", value<string>()->default_value("model"), 
        "base filename of model output files")
    ("lambda,r", value<float>()->default_value(0.0), 
        "regularisation strength parameter")
    ("dump-frequency", value<int>()->default_value(0), 
        "dump model every n iterations.")
    ("word-width", value<int>()->default_value(100), 
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("lbfgs-vectors", value<int>()->default_value(10), 
        "number of gradient history vectors for lbfgs.")
    ("test-tokens", value<int>()->default_value(10000), 
        "number of evenly space test points tokens evaluate.")
    ("gnorm-threshold", value<float>()->default_value(1.0), 
        "Terminat LBFGS iterations if the gradient norm falls below this value.")
    ("eta", value<float>()->default_value(0.00001), 
        "SGD eta, if used.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
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
  if (!vm.count("input")) { 
    cout << cmdline_options << "\n"; 
    return 1; 
  }

  ModelData config;
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
  cerr << "# lambda = " << vm["lambda"].as<float>() << endl;
  cerr << "# iterations = " << vm["iterations"].as<int>() << endl;
  cerr << "# threads = " << vm["threads"].as<int>() << endl;
  cerr << "# lbfgs history vectors = " << vm["lbfgs-vectors"].as<int>() << endl;
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

  VectorReal word_freq = VectorReal::Zero(model.labels());
  for (auto w : training_corpus)
    word_freq(w) += 1;
  if (!vm.count("model-in"))
    model.B = ((word_freq.array()+1.0)/(word_freq.sum()+word_freq.size())).log();

  int num_weights = model.num_weights();
  Real* gradient_data = new Real[num_weights];
  LogBiLinearModel::WeightsType gradient(gradient_data, num_weights);

//  int thread_id = omp_get_thread_num();
  Real lambda = config.l2_parameter; 

  Real f=0.0, wnorm=0.0, gnorm=numeric_limits<Real>::max();

  scitbx::lbfgs::minimizer<Real>* minimiser = new scitbx::lbfgs::minimizer<Real>(num_weights, vm["lbfgs-vectors"].as<int>());
  int function_evaluations=0;
  bool calc_g_and_f = true;
  int lbfgs_iteration = 0; //minimiser->iter();
  clock_t lbfgs_time=0, gradient_time=0;
/*
  lbfgs_t *opt = lbfgs_create(num_weights, 20, 0.001);
  while (lbfgs_iteration < vm["iterations"].as<int>() && gnorm > vm["gnorm-threshold"].as<float>()) {
    if (calc_g_and_f) {
      clock_t gradient_start = clock();

      gradient.setZero();
      f = function_and_gradient(model, training_corpus, lambda, gradient, gradient_data, wnorm, gnorm);
      function_evaluations++;
      gradient_time += (clock() - gradient_start);
    }

    cerr << "  (" << opt->niter << "." << opt->nfuns << ":" 
      << "f=" << f << ",|w|=" << model.W.norm() << ",|g|=" << gradient.norm();

    if (vm.count("test-set") && lbfgs_iteration != opt->niter) {
      lbfgs_iteration = opt->niter;
      cerr << ", Test Perplexity = " 
        << perplexity(model, test_corpus, test_corpus.size()/vm["test-tokens"].as<int>());
    }
    cerr << ")\n";

    clock_t lbfgs_start=clock();
    int iflag = lbfgs_run(opt, model.data(), &f, gradient_data);
    if (iflag < 0) {
      lbfgs_destory(opt);
      std::cerr << "\n\nlbfgs routine stops with an error" << std::endl;
      exit(1);
    }
    else if (iflag == 0) {
      std::cout << "\n\nConverged after " << lbfgs_iteration << " iterations." << std::endl;
      break;
    }
    lbfgs_time += (clock() - lbfgs_start);
  }
*/ 
/*
  while (lbfgs_iteration < vm["iterations"].as<int>() && gnorm > vm["gnorm-threshold"].as<float>()) {
    if (calc_g_and_f) {
      clock_t gradient_start = clock();

      gradient.setZero();
      f = function_and_gradient(model, training_corpus, lambda, gradient, gradient_data, wnorm, gnorm );
      function_evaluations++;
      gradient_time += (clock() - gradient_start);
    }

    if (lbfgs_iteration == 0 || (!calc_g_and_f )) {
      cout << "  (" << lbfgs_iteration+1 << "." << function_evaluations << ":" 
        << "f=" << f << ",|w|=" << wnorm << ",|g|=" << gnorm;
    }
    if (vm.count("test-set") && !calc_g_and_f)
      cout << ", Test Perplexity = " 
        << perplexity(model, test_corpus, test_corpus.size()/vm["test-tokens"].as<int>());
    if (lbfgs_iteration == 0 || (!calc_g_and_f ))
      cout << ")\n";

    clock_t lbfgs_start=clock();
    try { calc_g_and_f = minimiser->run(model.data(), f, gradient_data); }
    catch (const scitbx::lbfgs::error &e) {
      cerr << "LBFGS terminated with error:\n  " << e.what() << "\nRestarting..." << endl;
      delete minimiser;
      minimiser = new scitbx::lbfgs::minimizer<Real>(num_weights, vm["lbfgs-vectors"].as<int>());
      calc_g_and_f = true;
    }
    lbfgs_iteration = minimiser->iter();
    lbfgs_time += (clock() - lbfgs_start);
  }

  minimiser->run(model.data(), f, gradient_data);
  delete minimiser;
*/
  VectorReal adaGrad = gradient;
  Real eta = vm["eta"].as<float>();
  for (int lbfgs_iteration=0; lbfgs_iteration < vm["iterations"].as<int>(); ++lbfgs_iteration) {
      gradient.setZero();
      f = function_and_gradient(model, training_corpus, lambda, 
                                gradient, gradient_data, wnorm, gnorm);
      for (int g=0; g<num_weights; ++g) 
        adaGrad(g) += pow(gradient(g),2);

      function_evaluations++;
      cout << "  (" << lbfgs_iteration+1 << "." << function_evaluations << ":" 
        << "f=" << f << ",|w|=" << wnorm << ",|g|=" << gnorm;
      cout << ", Test Perplexity = " 
           << perplexity(model, test_corpus, test_corpus.size()/vm["test-tokens"].as<int>())
           << ")\n";

      //model.W = model.W - (0.0001*gradient);

      for (int w=0; w<num_weights; ++w){
        Real g = (adaGrad(w) == 0.0 ? 0.0 : eta / sqrt(adaGrad(w)));
        model.W(w) -= (g*gradient(w));
      }
  }

  if (vm.count("test-set"))
    cerr << "  Final Test Perplexity = " 
      << perplexity(model, test_corpus, test_corpus.size()/vm["test-tokens"].as<int>()) 
      << endl;

  if (vm.count("model-out")) {
    cout << "Writing trained model to " << vm["model-out"].as<string>() << endl;
    std::ofstream f(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f);
    ar << model;
  }

  delete gradient_data;
}


Real function_and_gradient(LogBiLinearModel& model, const Corpus& training_corpus,
                           Real lambda, LogBiLinearModel::WeightsType& gradient,
                           Real* gradient_data, Real& wnorm, Real& gnorm) {
  cerr << "Function_and_gradient" << endl;
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;
  int num_words = model.labels();

  std::vector<VectorReal> log_partition_functions(context_width, VectorReal(num_words));
  std::vector<VectorReal> context_expectations(context_width, VectorReal::Zero(num_words));

  // cache the mixture weights
  VectorReal pM = softMax(model.M);

  // cache the products of Q with the contexts 
  cerr << "  - caching " << num_words << " Q products" << endl;

  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i < context_width; i++)
    q_context_products.at(i) = model.Q * model.C.at(i);

  cerr << "  - caching Z(w)";
  boost::progress_display show_progress(context_width*num_words, std::cerr, "\n  ", "  ", "  ");

  // cache the partition functions
  #pragma omp parallel 
  { 

    boost::timer timer;
    #pragma omp for 
    for (int q=0; q < num_words; ++q) { // O(|v|^2)
      for (int i=0; i < context_width; i++) { // O(context_width * |v|^2)
        VectorReal logProbs = model.R * q_context_products[i].row(q).transpose() + model.B; 
        Real max_logProb = logProbs.maxCoeff();
        Real logProbs_z = log((logProbs.array() - max_logProb).exp().sum()) + max_logProb;

        assert(isfinite(logProbs_z));
        log_partition_functions[i](q) = logProbs_z;

        //if (tid==0) 
        #pragma omp critical
        ++show_progress;
      }
    }
    #pragma omp master
    cerr << timer.elapsed() << " seconds." << endl;

    Real local_f = 0;
//    int tid = omp_get_thread_num();
    int R_size = num_words*word_width;
    int Q_size = R_size;
    int C_size = word_width*word_width;
    int B_size = num_words;
    int M_size = context_width;

    assert((R_size+Q_size+context_width*C_size+B_size+M_size) == model.num_weights());
    Real* thread_local_gradient_data = new Real[model.num_weights()];

    LogBiLinearModel::WordVectorsType g_R(thread_local_gradient_data, num_words, word_width);
    LogBiLinearModel::WordVectorsType g_Q(thread_local_gradient_data+R_size, num_words, word_width);

    LogBiLinearModel::ContextTransformsType g_C;
    Real* ptr = thread_local_gradient_data+2*R_size;
    for (int i=0; i<context_width; i++) {
      g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, word_width));
      ptr += C_size;
    }

    LogBiLinearModel::WeightsType g_B(ptr, num_words);
    LogBiLinearModel::WeightsType g_M(ptr+num_words, context_width);

    // data update component
    #pragma omp master 
    {
      cerr << "  - data update";
      show_progress.restart(training_corpus.size()*context_width);
    }
    timer.restart();

    #pragma omp for
    for (size_t t=0; t < training_corpus.size(); ++t) {
      WordId w = training_corpus.at(t);
      int context_start = t - context_width;

      VectorReal logProbs(context_width);
      // calculate the log-probabilities
      for (int i=0; i<context_width; i++) {
        int j=context_start+i;
        int q = (j<0 ? start_id : training_corpus.at(j));
        logProbs(i) = exp(model.R.row(w) * q_context_products[i].row(q).transpose() + model.B(w)
                          - log_partition_functions.at(i)(q));
        logProbs(i) *= pM(i);
      }
      local_f -= log(logProbs.sum());

      // calculate the mixture contributions
      Real contributions_z = logProbs.sum();
      VectorReal contributions = logProbs.array()/contributions_z;

      // do the data gradient update
      for (int i=0; i<context_width; i++) {
        int j=context_start+i;
        int q = (j<0 ? start_id : training_corpus.at(j));

        context_expectations.at(i)(q) += contributions(i);

        // data expectations
        g_R.row(w) -= (contributions(i) * q_context_products[i].row(q));
        g_Q.row(q) -= (contributions(i) * (model.C.at(i) * model.R.row(w).transpose()));
        g_C.at(i)  -= (contributions(i) * (model.Q.row(q).transpose() * model.R.row(w)));
        g_B(w)     -=  contributions(i);

        g_M(i)     -=  (contributions(i) - pM(i));

        //      if (t % 100000 == 0) {cerr << "."; cerr.flush(); }
        //if (tid==0) ++show_progress;
        #pragma omp critical
        ++show_progress;
      }
    }
    #pragma omp master
    cerr << timer.elapsed() << " seconds." << endl;
    //  cerr << "g_R:\n" << g_R << endl;
    //  cerr << "g_Q:\n" << g_Q << endl;
    //  cerr << "g_C:\n";
    //  for (auto C : g_C)
    //    cerr << C << endl;

    // model update component and
    #pragma omp master
    {
      cerr << "  - model update";
      show_progress.restart(context_width*num_words);
    }

    timer.restart();

    #pragma omp for 
    for (int q=0; q < num_words; ++q) { // O(context_width * |v|^2)
      for (int i=0; i < context_width; i++) { // O(context_width)
//        VectorReal logProbs = model.R * q_context_products[i].row(q).transpose() + model.B; 
//        VectorReal probs = (logProbs.array() - log_partition_functions[i](q)).exp();

        VectorReal probs = ((model.R*q_context_products[i].row(q).transpose()+model.B).array() - log_partition_functions[i](q)).exp();
        VectorReal expected_representation = (probs.transpose() * model.R); // O(|v|*word_width)

        // model expectations
        Real q_freq = context_expectations[i](q);

        // something fishy about the Eigen outer produce. Explicit column wise loop is twice as fast.
        // However both are unfriendly cache behavior for openmp.
        //g_R        += (q_freq*probs) * q_context_products[i].row(q); // O(|v|*word_width)
        VectorReal freq_probs = q_freq*probs;
        for (int w=0; w<word_width; ++w)
          g_R.col(w) += freq_probs*q_context_products[i].row(q)(w); // O(|v|*word_width)

        g_Q.row(q) += (q_freq * (model.C.at(i) * expected_representation).transpose());// O(word_width^2)
        g_C.at(i)  += (q_freq * (model.Q.row(q).transpose() * expected_representation.transpose()));// O(word_width^2)
        g_B        += freq_probs;// O(|v|)

        #pragma omp critical
        ++show_progress;
      }
    }

    #pragma omp master
    cerr << timer.elapsed() << " seconds." << endl;
    //  cerr << "g_R:\n" << g_R << endl;
    //  cerr << "g_Q:\n" << g_Q << endl;
    //  cerr << "g_C:\n";
    //  for (auto C : g_C)
    //    cerr << C << endl;

    // synchronise the gradients
    #pragma omp critical 
    {
      f += local_f;
      for (int i=0; i<model.num_weights(); ++i)
        gradient_data[i] += thread_local_gradient_data[i];
    }

    #pragma omp master
    {
      cerr << "Model M: " << pM.transpose() << endl;
      cerr << "g_M:     " << g_M.transpose() << endl;
      cerr << "f=" << f << endl;
    }
  }
  wnorm = model.W.norm();
  gnorm = gradient.norm();

  return f;
}


Real perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;

  // cache the products of Q with the contexts 
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.Q * model.C.at(i);

  // cache the mixture weights
  VectorReal pM = softMax(model.M);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  #pragma omp parallel \
      shared(test_corpus,model,stride,q_context_products) \
      reduction(+:p,tokens) 
  {
    size_t thread_num = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    for (size_t s = (thread_num*stride); s < test_corpus.size(); s += (num_threads*stride)) {
      WordId w = test_corpus.at(s);

      int context_start = s - context_width;
      Real p_sum = 0;
      for (int i=0; i<context_width; ++i) {
        int j=context_start+i;
        int v_i = (j<0 ? start_id : test_corpus.at(j));

        ArrayReal score_vector = model.R * q_context_products[i].row(v_i).transpose() + model.B;
        p_sum += (pM(i) * softMax(score_vector)(w));
      }
      
      p += log(p_sum);
      tokens++;
    }
  }
  return exp(-p/tokens);
}
