// STL
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iterator>
#include <cstring>
#include <functional>
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
#include "corpus/corpus.h"
#include "lbl/log_add.h"
#include "lbl/log_bilinear_model.h"
#include "utils/conditional_omp.h"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;


typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

void learn(const ModelData& config);

Real sgd_update(LogBiLinearModel& model, MatrixReal& class_word_probs, MatrixReal& class_word_statistics,
                int start, int end, const Corpus& training_corpus,
                const vector<size_t> &indices, Real lambda, Real step_size, Real multinomial_eta);

Real perplexity(const LogBiLinearModel& model, const MatrixReal& class_word_statistics,
                const Corpus& test_corpus, int stride=1);


int main(int argc, char **argv) {
  cout << "A class based log-bilinear language model: Copyright 2013 Phil Blunsom, "
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
    ("minibatch-size", value<int>()->default_value(1000),
        "number of sentences per minibatch")
    ("order,n", value<int>()->default_value(3),
        "ngram order")
    ("model-in,m", value<string>(),
        "initial model")
    ("model-out,o", value<string>()->default_value("model"),
        "base filename of model output files")
    ("lambda,r", value<float>()->default_value(0.0),
        "regularisation strength parameter")
    ("classes,c", value<int>()->default_value(50),
        "number of latent classes.")
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("test-tokens", value<int>()->default_value(10000),
        "number of evenly space test points tokens evaluate.")
    ("step-size", value<float>()->default_value(0.001),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("multinomial-step-size", value<float>()->default_value(0.1),
        "Online EM step-size for p(w|c) distributions.")
    ("randomise", "visit the training tokens in random order");
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
  config.training_file = vm["input"].as<string>();
  if (vm.count("test-set")) {
    config.test_file = vm["test-set"].as<string>();
  }
  config.iterations = vm["iterations"].as<int>();
  config.minibatch_size = vm["minibatch-size"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  if (vm.count("model-in")) {
    config.model_input_file = vm["model-in"].as<string>();
  }
  if (vm.count("model-out")) {
    config.model_output_file = vm["model-out"].as<string>();
  }
  config.l2_parameter = vm["lambda"].as<float>();
  config.classes = vm["classes"].as<int>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.classes =  vm["classes"].as<int>();
  config.test_tokens = vm["test-tokens"].as<int>();
  config.step_size = vm["step-size"].as<float>();
  config.multinomial_step_size = vm["multinomial-step-size"].as<float>();
  config.randomise = vm.count("randomise");

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  cerr << "# order = " << config.ngram_order << endl;
  if (config.model_input_file.size()) {
    cerr << "# model-in = " << config.model_input_file << endl;
  }
  if (config.model_output_file.size()) {
    cerr << "# model-out = " << config.model_output_file << endl;
  }
  cerr << "# input = " << config.training_file << endl;
  cerr << "# minibatch-size = " << config.minibatch_size << endl;
  cerr << "# lambda = " << config.l2_parameter << endl;
  cerr << "# classes = " << config.classes << endl;
  cerr << "# iterations = " << config.iterations << endl;
  cerr << "# threads = " << config.threads << endl;
  cerr << "################################" << endl;

  omp_set_num_threads(config.threads);

  learn(config);

  return 0;
}


void learn(const ModelData& config) {
  Corpus training_corpus, test_corpus;

  //////////////////////////////////////////////
  // read the training sentences
  ifstream in(config.training_file);
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
  if (config.test_file.size()) {
    ifstream test_in(config.test_file);
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

  Dict classes_dict;
  for (int c=0; c < config.classes; ++c)
    classes_dict.Convert("C"+std::to_string(c));

  LogBiLinearModel model(config, dict);

  if (config.model_input_file.size()) {
    std::ifstream f(config.model_input_file);
    boost::archive::text_iarchive ar(f);
    ar >> model;

    if (static_cast<size_t>(model.labels()) != classes_dict.size()) {
      cerr << "Input model in file: " << config.model_input_file << " doesn't have "
           << classes_dict.size() << "classes." << endl;
      exit(1);
    }
  }

  vector<size_t> training_indices(training_corpus.size());
  MatrixReal class_word_statistics = MatrixReal::Identity(dict.size(), model.output_types());
  {
    VectorReal unigram = VectorReal::Zero(dict.size());
    for (size_t i=0; i<training_indices.size(); i++) {
      unigram(training_corpus[i]) += 1;
      training_indices[i] = i;
    }
    unigram /= unigram.sum();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.01);
    for (int i=0; i<class_word_statistics.rows(); i++)
      for (int j=0; j<class_word_statistics.cols(); j++)
        class_word_statistics(i,j) = unigram(i) + std::abs(gaussian(gen));
  }

  size_t minibatch_counter=0;
  size_t minibatch_size = config.minibatch_size;
  MatrixReal class_word_probs = MatrixReal::Zero(dict.size(), model.output_types());
  for (int iteration=0; iteration < config.iterations; ++iteration) {
    clock_t iteration_start=clock();
    Real av_f=0.0;
    cout << "Iteration " << iteration << ": "; cout.flush();

    if (config.randomise)
      std::random_shuffle(training_indices.begin(), training_indices.end());

    // normalise the multinomials
//    class_word_statistics.array() += 0.5;
//    for (int i=0; i<class_word_probs.cols(); i++) {
//      class_word_probs.col(i) = class_word_statistics.col(i) / class_word_statistics.col(i).sum();
//      assert(class_word_statistics.col(i).sum() > 0);
//    }

//    class_word_statistics.setZero();

    #pragma omp parallel \
      firstprivate(minibatch_counter) \
      shared(training_corpus,training_indices,model,vm,minibatch_size,iteration,config,\
             class_word_statistics,class_word_probs) \
      reduction(+:av_f)
    {
      int thread_id = omp_get_thread_num();
      int thread_minibatch_size = minibatch_size / omp_get_num_threads();
      for (size_t start=thread_id*thread_minibatch_size; start < training_corpus.size(); ++minibatch_counter) {
        size_t end = min(training_corpus.size(), start + thread_minibatch_size);

        Real lambda = config.l2_parameter*(end-start)/static_cast<Real>(training_corpus.size());
        Real f=0.0;
        Real step_size = config.step_size;
        f = sgd_update(model, class_word_probs, class_word_statistics, start, end, training_corpus,
                       training_indices, lambda, step_size, config.multinomial_step_size);
        av_f += f;

        // regularisation
        if (lambda > 0) model.l2_gradient_update(step_size*lambda);

        #pragma omp master
        if (minibatch_counter % 100 == 0) { cerr << "."; cout.flush(); }

        start += (thread_minibatch_size*omp_get_num_threads());
      }
    }

    Real iteration_time = (clock()-iteration_start) / (Real)CLOCKS_PER_SEC;
    cerr << "\n | Time: " << iteration_time << " seconds, Average f = " << av_f/training_corpus.size();
    if (test_corpus.size()) {
      cerr << ", Test Perplexity = "
           << perplexity(model, class_word_probs, test_corpus, test_corpus.size() / config.test_tokens);
    }
    cerr << " |" << endl << endl;
  }
  class_word_probs = class_word_statistics;
  for (int i=0; i<class_word_probs.cols(); i++)
    class_word_probs.col(i) = class_word_probs.col(i) / class_word_probs.col(i).sum();

  if (config.model_output_file.size()) {
    cout << "Writing trained model to " << config.model_output_file << endl;
    std::ofstream f(config.model_output_file);
    boost::archive::text_oarchive ar(f);
    ar << model;
  }
}


Real sgd_update(LogBiLinearModel& model, MatrixReal& class_word_probs, MatrixReal& class_word_statistics,
                int start, int end, const Corpus& training_corpus,
                const vector<size_t> &training_indices,
                Real lambda, Real step_size, Real multinomial_eta) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // normalise the multinomials
//  MatrixReal class_word_probs = class_word_statistics;
  for (int i=0; i<class_word_probs.cols(); i++)
    class_word_probs.col(i) = class_word_statistics.col(i) / class_word_statistics.col(i).sum();

  // form matrices of the ngram histories
//  clock_t cache_start = clock();
  int instances=end-start;
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width));
  for (int instance=0; instance < instances; ++instance) {
    int w_i = training_indices.at(instance);

    int context_start = w_i - context_width;
    for (int i=0; i<context_width; i++) {
      int j=context_start+i;
      int v_i = (j<0 ? start_id : training_corpus.at(j));
      context_vectors.at(i).row(instance) = model.Q.row(v_i);
    }
  }
  MatrixReal prediction_vectors = MatrixReal::Zero(instances, word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += (context_vectors.at(i) * model.C.at(i));
//  clock_t cache_time = clock() - cache_start;

  // calculate the weight sum of word representations
  MatrixReal weightedRepresentations = MatrixReal::Zero(instances, word_width);

  // calculate the function and gradient for each ngram
//  clock_t iteration_start = clock();
  ArrayReal class_conditional_probs;
  MatrixReal class_marginal_probs(instances, model.output_types());
  MatrixReal class_probs_delta(instances, model.output_types());
  for (int instance=0; instance < instances; instance++) {
    int w_i = training_indices.at(instance);
    WordId w = training_corpus.at(w_i);

    VectorReal class_conditional_log_probs = model.R * prediction_vectors.row(instance).transpose() + model.B;
    Real max_log_prob = class_conditional_log_probs.maxCoeff();
    Real class_conditional_probs_log_z = log((class_conditional_log_probs.array() - max_log_prob).exp().sum())
                                         + max_log_prob;
    assert(isfinite(class_conditional_probs_log_z));

    class_conditional_log_probs.array() -= class_conditional_probs_log_z;

    // model update component
    class_conditional_probs = class_conditional_log_probs.array().exp();

    // data update component
    class_marginal_probs.row(instance) = class_conditional_probs.transpose() * class_word_probs.row(w).array();

    assert(isfinite(class_marginal_probs.row(instance).sum()));
    assert(isfinite(log(class_marginal_probs.row(instance).sum())));
    f += log(class_marginal_probs.row(instance).sum());

    class_marginal_probs.row(instance).array() /= class_marginal_probs.row(instance).sum();

    class_probs_delta.row(instance) = class_conditional_probs.transpose() - class_marginal_probs.row(instance).array();
    weightedRepresentations.row(instance) = class_probs_delta.row(instance) * model.R;
  }

  // do the gradient updates
  class_word_statistics.array() *= (1.0f - multinomial_eta);
  for (int instance=0; instance < instances; instance++) {
    int w_i = training_indices.at(instance);
    WordId w = training_corpus.at(w_i);

    // LBL-LM output updates
    model.R -= step_size * class_probs_delta.row(instance).transpose() * prediction_vectors.row(instance);
    model.B -= step_size * class_probs_delta.row(instance);

    // multinomial online EM update
    class_word_statistics.row(w) += multinomial_eta * class_marginal_probs.row(instance);
  }
//  clock_t iteration_time = clock() - iteration_start;

//  clock_t context_start = clock();
  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i=0; i<context_width; ++i) {
    context_gradients = (model.C.at(i) * weightedRepresentations.transpose()).transpose();
    for (int instance=0; instance < instances; ++instance) {
      int j = training_indices.at(instance) - context_width + i;
      int v_i = (j<0 ? start_id : training_corpus.at(j));
      model.Q.row(v_i) -= step_size * context_gradients.row(instance);
    }
    model.C.at(i) -= step_size * context_vectors.at(i).transpose() * weightedRepresentations;
  }
//  clock_t context_time = clock() - context_start;

  return f;
}


Real perplexity(const LogBiLinearModel& model, const MatrixReal& class_word_statistics,
                const Corpus& test_corpus, int stride) {
  Real p=0.0;

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  MatrixReal class_word_probs = class_word_statistics;
  for (int i=0; i<class_word_probs.cols(); i++)
    class_word_probs.col(i) = class_word_probs.col(i) / class_word_probs.col(i).sum();

  // cache the products of Q with the contexts
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.Q * model.C.at(i);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  #pragma omp parallel \
      shared(test_corpus,model,stride,q_context_products,word_width) \
      reduction(+:p,tokens)
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

      // unnormalised log p(c | context)
      VectorReal score_vector = model.R * prediction_vector + model.B;

      // get the log normaliser, avoiding overflow
      Real max_score = score_vector.maxCoeff();
      Real log_z = log((score_vector.array()-max_score).exp().sum()) + max_score;

      // marginalise out the class to get p(w | context)
      Real w_prob = class_word_probs.row(w) * (score_vector.array()-log_z).exp().matrix();

      // update the accumulators
      p += log(w_prob);
      tokens++;
    }
  }
  cerr << "\n" << p << " " << -p/tokens << endl;
  return exp(-p/tokens);
}
