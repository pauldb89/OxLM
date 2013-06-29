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

void learn(const variables_map& vm, const ModelData& config);

typedef int TrainingInstance;
typedef vector<TrainingInstance> TrainingInstances;
void cache_data(int start, int end, 
                const Corpus& training_corpus, 
                const vector<size_t>& indices,
                TrainingInstances &result);

Real sgd_gradient(FactoredOutputLogBiLinearModel& model,
                const Corpus& training_corpus, 
                const TrainingInstances &indexes,
                Real lambda, 
                LogBiLinearModel::WordVectorsType& g_R,
                LogBiLinearModel::WordVectorsType& g_Q,
                LogBiLinearModel::ContextTransformsType& g_C,
                LogBiLinearModel::WeightsType& g_B,
                MatrixReal & g_F);


Real perplexity(const FactoredOutputLogBiLinearModel& model, const Corpus& test_corpus, int stride=1);
void freq_bin_type(const std::string &corpus, int num_classes, std::vector<int>& classes, Dict& dict);


int main(int argc, char **argv) {
  cout << "Online noise contrastive estimation for log-bilinear models: Copyright 2013 Phil Blunsom, " 
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
    ("instances", value<int>()->default_value(std::numeric_limits<int>::max()), 
        "training instances per iteration")
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
    ("word-width", value<int>()->default_value(100), 
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("test-tokens", value<int>()->default_value(10000), 
        "number of evenly spaced test points tokens evaluate.")
    ("step-size", value<float>()->default_value(1.0), 
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("classes", value<int>()->default_value(100), 
        "number of classes for factored output.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
    ("randomise", "visit the training tokens in random order")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
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
  config.l2_parameter = vm["lambda"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  config.verbose = vm.count("verbose");
  config.classes = vm["classes"].as<int>();

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  cerr << "# order = " << vm["order"].as<int>() << endl;
  if (vm.count("model-in"))
    cerr << "# model-in = " << vm["model-in"].as<string>() << endl;
  cerr << "# model-out = " << vm["model-out"].as<string>() << endl;
  cerr << "# input = " << vm["input"].as<string>() << endl;
  cerr << "# minibatch-size = " << vm["minibatch-size"].as<int>() << endl;
  cerr << "# lambda = " << vm["lambda"].as<float>() << endl;
  cerr << "# iterations = " << vm["iterations"].as<int>() << endl;
  cerr << "# threads = " << vm["threads"].as<int>() << endl;
  cerr << "# classes = " << config.classes << endl;
  cerr << "################################" << endl;

  omp_set_num_threads(config.threads);

  learn(vm, config);

  return 0;
}


void learn(const variables_map& vm, const ModelData& config) {
  Corpus training_corpus, test_corpus;
  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");

  //////////////////////////////////////////////
  // separate the word types into classes using
  // frequency binning
  vector<int> classes;
  freq_bin_type(vm["input"].as<string>(), vm["classes"].as<int>(), classes, dict);
  //////////////////////////////////////////////


  //////////////////////////////////////////////
  // read the training sentences
  ifstream in(vm["input"].as<string>().c_str());
  string line, token;

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
      while (line_stream >> token) {
        WordId w = dict.Convert(token, true);
        if (w < 0) {
          cerr << token << " " << w << endl;
          assert(!"Unknown word found in test corpus.");
        }
        test_corpus.push_back(w);
      }
      test_corpus.push_back(end_id);
    }
    test_in.close();
  }
  //////////////////////////////////////////////

  //LogBiLinearModel model(config, dict, vm.count("diagonal-contexts"));
  FactoredOutputLogBiLinearModel model(config, dict, vm.count("diagonal-contexts"), classes);

  if (vm.count("model-in")) {
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;
  }

  vector<size_t> training_indices(training_corpus.size());
  model.unigram = VectorReal::Zero(model.labels());
  for (size_t i=0; i<training_indices.size(); i++) {
    model.unigram(training_corpus[i]) += 1;
    training_indices[i] = i;
  }
  model.B = ((model.unigram.array()+1.0)/(model.unigram.sum()+model.unigram.size())).log();
  model.unigram /= model.unigram.sum();

  VectorReal adaGrad = VectorReal::Zero(model.num_weights());
  VectorReal global_gradient(model.num_weights());
  Real av_f=0.0;
  Real pp=0;

  MatrixReal global_gradientF(model.F.rows(), model.F.cols());
  MatrixReal adaGradF = MatrixReal::Zero(model.F.rows(), model.F.cols());

  #pragma omp parallel shared(global_gradient, global_gradientF)
  {
    //////////////////////////////////////////////
    // setup the gradient matrices
    int num_words = model.labels();
    int num_classes = model.config.classes;
    int word_width = model.config.word_representation_size;
    int context_width = model.config.ngram_order-1;

    int R_size = num_words*word_width;
    int Q_size = R_size;
    int C_size = (vm.count("diagonal-contexts") ? word_width : word_width*word_width);
    int B_size = num_words;
    int M_size = context_width;

    assert((R_size+Q_size+context_width*C_size+B_size+M_size) == model.num_weights());

    Real* gradient_data = new Real[model.num_weights()];
    LogBiLinearModel::WeightsType gradient(gradient_data, model.num_weights());

    LogBiLinearModel::WordVectorsType g_R(gradient_data, num_words, word_width);
    LogBiLinearModel::WordVectorsType g_Q(gradient_data+R_size, num_words, word_width);

    LogBiLinearModel::ContextTransformsType g_C;
    Real* ptr = gradient_data+2*R_size;
    for (int i=0; i<context_width; i++) {
      if (vm.count("diagonal-contexts"))
          g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, 1));
      else
          g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, word_width));
      ptr += C_size;
    }

    LogBiLinearModel::WeightsType g_B(ptr, B_size);
    LogBiLinearModel::WeightsType g_M(ptr+B_size, M_size);
    MatrixReal g_F(num_classes, word_width);
    //////////////////////////////////////////////

    size_t minibatch_counter=0;
    size_t minibatch_size = vm["minibatch-size"].as<int>();
    for (int iteration=0; iteration < vm["iterations"].as<int>(); ++iteration) {
      clock_t iteration_start=clock();
      #pragma omp master
      {
        av_f=0.0;
        pp=0.0;
        cout << "Iteration " << iteration << ": "; cout.flush();

        if (vm.count("randomise"))
          std::random_shuffle(training_indices.begin(), training_indices.end());
      }

      TrainingInstances training_instances;
      Real step_size = vm["step-size"].as<float>(); //* minibatch_size / training_corpus.size();

      for (size_t start=0; start < training_corpus.size() && (int)start < vm["instances"].as<int>(); ++minibatch_counter) {
        size_t end = min(training_corpus.size(), start + minibatch_size);

        #pragma omp master
        {
          global_gradient.setZero();
          global_gradientF.setZero();
        }

        gradient.setZero();
        g_F.setZero();
        Real lambda = config.l2_parameter*(end-start)/static_cast<Real>(training_corpus.size()); 

        #pragma omp barrier
        cache_data(start, end, training_corpus, training_indices, training_instances);
        Real f = sgd_gradient(model, training_corpus, training_instances, lambda, g_R, g_Q, g_C, g_B, g_F);

        #pragma omp critical 
        {
          global_gradient += gradient;
          global_gradientF += g_F;
          av_f += f;
        }
        #pragma omp barrier 
        #pragma omp master
        {
          adaGrad.array() += global_gradient.array().square();
          for (int w=0; w<model.num_weights(); ++w)
            if (adaGrad(w)) model.W(w) -= (step_size*global_gradient(w) / sqrt(adaGrad(w)));

          adaGradF.array() += global_gradientF.array().square();
          for (int r=0; r < adaGradF.rows(); ++r)
            for (int c=0; c < adaGradF.cols(); ++c)
              if (adaGradF(r,c)) model.F(r,c) -= (step_size*global_gradientF(r,c) / sqrt(adaGradF(r,c)));

          // regularisation
          if (lambda > 0) av_f += (0.5*lambda*model.l2_gradient_update(step_size*lambda));

          if (minibatch_counter % 100 == 0) { cerr << "."; cout.flush(); }
        }

        //start += (minibatch_size*omp_get_num_threads());
        start += minibatch_size;
      }
      #pragma omp master
      cerr << endl;

      Real iteration_time = (clock()-iteration_start) / (Real)CLOCKS_PER_SEC;
      if (vm.count("test-set")) {
        Real local_pp = perplexity(model, test_corpus, 1);

        #pragma omp critical 
        { pp += local_pp; }
        #pragma omp barrier
      }

      #pragma omp master
      {
        pp = exp(-pp/test_corpus.size());
        cerr << " | Time: " << iteration_time << " seconds, Average f = " << av_f/training_corpus.size();
        if (vm.count("test-set")) {
          cerr << ", Test Perplexity = " << pp; 
        }
        if (vm.count("mixture"))
          cerr << ", Mixture weights = " << softMax(model.M).transpose();
        cerr << " |" << endl << endl;
      }
    }
  }

  if (vm.count("model-out")) {
    cout << "Writing trained model to " << vm["model-out"].as<string>() << endl;
    std::ofstream f(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f);
    ar << model;
  }
}


void cache_data(int start, int end, const Corpus& training_corpus, const vector<size_t>& indices, TrainingInstances &result) {
  assert (start>=0 && start < end && end <= static_cast<int>(training_corpus.size()));
  assert (training_corpus.size() == indices.size());

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  result.clear();
  result.reserve((end-start)/num_threads);

  for (int s = start+thread_num; s < end; s += num_threads) {
    result.push_back(indices.at(s));
  }
}


Real sgd_gradient(FactoredOutputLogBiLinearModel& model,
                const Corpus& training_corpus,
                const TrainingInstances &training_instances,
                Real lambda, 
                LogBiLinearModel::WordVectorsType& g_R,
                LogBiLinearModel::WordVectorsType& g_Q,
                LogBiLinearModel::ContextTransformsType& g_C,
                LogBiLinearModel::WeightsType& g_B,
                MatrixReal& g_F) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");
  WordId end_id = model.label_set().Convert("</s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // form matrices of the ngram histories
//  clock_t cache_start = clock();
  int instances=training_instances.size();
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width)); 
  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    int context_start = t - context_width;

    bool sentence_start = (t==0);
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || training_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : training_corpus.at(j));
      context_vectors.at(i).row(instance) = model.Q.row(v_i);
    }
  }
  MatrixReal prediction_vectors = MatrixReal::Zero(instances, word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += model.context_product(i, context_vectors.at(i));

//  clock_t cache_time = clock() - cache_start;

  // calculate the weight sum of word representations
  MatrixReal weightedRepresentations = MatrixReal::Zero(instances, word_width);

  // calculate the function and gradient for each ngram
//  clock_t iteration_start = clock();
  for (int instance=0; instance < instances; instance++) {
    int w_i = training_instances.at(instance);
    WordId w = training_corpus.at(w_i);
    int c = model.get_class(w);
    int c_start = model.indexes.at(c), c_end = model.indexes.at(c+1);

    VectorReal class_conditional_scores = model.F * prediction_vectors.row(instance).transpose();// + model.B;

    VectorReal word_conditional_scores
      = model.class_R(c) * prediction_vectors.row(instance).transpose() + model.class_B(c);

    ArrayReal class_conditional_log_probs = logSoftMax(class_conditional_scores);
    ArrayReal word_conditional_log_probs  = logSoftMax(word_conditional_scores);

    VectorReal class_conditional_probs = class_conditional_log_probs.exp();
    VectorReal word_conditional_probs  = word_conditional_log_probs.exp();

    weightedRepresentations.row(instance) -= (model.F.row(c) - class_conditional_probs.transpose() * model.F);
    weightedRepresentations.row(instance) -= (model.R.row(w) - word_conditional_probs.transpose() * model.class_R(c));

    if (!isfinite(class_conditional_log_probs(c))) {
      cerr << c << " " << class_conditional_probs(c) << " \n" << class_conditional_scores.transpose() << endl;
      assert(false);
    }
    assert(isfinite(word_conditional_log_probs(w-c_start)));
    f -= (class_conditional_log_probs(c) + word_conditional_log_probs(w-c_start));

    // do the gradient updates:
    //   data contributions: 
    g_F.row(c) -= prediction_vectors.row(instance).transpose();
    g_R.row(w) -= prediction_vectors.row(instance).transpose();
    g_B(w)     -= word_conditional_probs(w-c_start);

    //   model contributions: 
    g_R.block(c_start, 0, c_end-c_start, g_R.cols()) 
        += word_conditional_probs * prediction_vectors.row(instance);
    g_F += class_conditional_probs * prediction_vectors.row(instance);
    g_B.segment(c_start, c_end-c_start) += word_conditional_probs;
    
  }
//  clock_t iteration_time = clock() - iteration_start;

//  clock_t context_start = clock();
  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i=0; i<context_width; ++i) {
    context_gradients = model.context_product(i, weightedRepresentations, true); // weightedRepresentations*C(i)^T

    for (int instance=0; instance < instances; ++instance) {
      int w_i = training_instances.at(instance);
      int j = w_i-context_width+i;

      bool sentence_start = (j<0);
      for (int k=j; !sentence_start && k < w_i; k++)
        if (training_corpus.at(k) == end_id) 
          sentence_start=true;
      int v_i = (sentence_start ? start_id : training_corpus.at(j));

      g_Q.row(v_i) += context_gradients.row(instance);
    }
    model.context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations);
  }
//  clock_t context_time = clock() - context_start;

  return f;
}


Real perplexity(const FactoredOutputLogBiLinearModel& model, const Corpus& test_corpus, int stride) {
  Real p=0.0;

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // cache the products of Q with the contexts 
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.context_product(i, model.Q);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

  {
    #pragma omp master
    cerr << "Calculating perplexity for " << test_corpus.size()/stride << " tokens";

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

      int c = model.get_class(w);
      int c_start = model.indexes.at(c);
      VectorReal class_probs = logSoftMax(model.F * prediction_vector);// + model.B;
      VectorReal word_probs = logSoftMax(model.class_R(c) * prediction_vector + model.class_B(c));
//      cerr << model.label_str(w) << ": class=" << c << " log_prob=" << class_probs(c) 
//           << "+" << word_probs(w-c_start) << " sum=" << word_probs.array().exp().sum() << endl;

      p += class_probs(c) + word_probs(w-c_start);

      #pragma omp master
      if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }

      tokens++;
    }
    #pragma omp master
    cerr << endl;
  }

  return p;
}


void freq_bin_type(const std::string &corpus, int num_classes, vector<int>& classes, Dict& dict) {
  ifstream in(corpus.c_str());
  string line, token;

  map<string,int> tmp_dict;
  vector< pair<string,int> > counts;
  int sum=0;
  string eos = "</s>";

  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token) {
      if (token == eos) continue;
      int w_id = tmp_dict.insert(make_pair(token,tmp_dict.size())).first->second;
      assert (w_id <= int(counts.size()));
      if (w_id == int(counts.size())) counts.push_back( make_pair(token, 1) );
      else                            counts[w_id].second += 1;
      sum++;
    }
  }

  sort(counts.begin(), counts.end(), 
       [](const pair<string,int>& a, const pair<string,int>& b) -> bool { return a.second > b.second; });

  classes.clear();
  classes.push_back(0);
  classes.push_back(2);

  int bin_size = sum / (num_classes-1);
  int mass=0;
  for (int i=0; i < int(counts.size()); ++i) {
    WordId id = dict.Convert(counts.at(i).first);
    if ((mass += counts.at(i).second) > bin_size) {
//      cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
      bin_size = (sum -= mass) / (num_classes - classes.size());

      classes.push_back(id+1);
      mass=0;
    }
  }
  if (classes.back() != int(dict.size()))
    classes.push_back(dict.size());
//  cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;

  cerr << "Binned " << dict.size() << " types in " << classes.size()-1 << " classes with an average of " 
       << float(dict.size()) / float(classes.size()-1) << " types per bin." << endl; 
  in.close();
}
