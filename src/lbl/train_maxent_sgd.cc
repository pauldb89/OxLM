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
#include <boost/lexical_cast.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "lbl/feature.h"
#include "lbl/feature_generator.h"
#include "lbl/feature_store.h"
#include "lbl/nlm.h"
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

void learn(const variables_map& vm, ModelData& config);

typedef int TrainingInstance;
typedef vector<TrainingInstance> TrainingInstances;
void cache_data(int start, int end,
                const Corpus& training_corpus,
                const vector<size_t>& indices,
                TrainingInstances &result);

Real sgd_gradient(FactoredMaxentNLM& model,
                  const Corpus& training_corpus,
                  const TrainingInstances &indexes,
                  Real lambda,
                  NLM::WordVectorsType& g_R,
                  NLM::WordVectorsType& g_Q,
                  NLM::ContextTransformsType& g_C,
                  NLM::WeightsType& g_B,
                  MatrixReal & g_F,
                  VectorReal & g_FB,
                  UnconstrainedFeatureStore& g_U,
                  vector<UnconstrainedFeatureStore>& g_V);


Real perplexity(const FactoredMaxentNLM& model, const Corpus& test_corpus);
void freq_bin_type(const std::string &corpus, int num_classes, std::vector<int>& classes, Dict& dict, VectorReal& class_bias);
void classes_from_file(const std::string &class_file, vector<int>& classes, Dict& dict, VectorReal& class_bias);


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
    ("order,n", value<int>()->default_value(5),
        "ngram order")
    ("feature-context-size", value<unsigned int>()->default_value(5),
        "size of the window for maximum entropy features")
    ("model-in,m", value<string>(),
        "initial model")
    ("model-out,o", value<string>()->default_value("model"),
        "base filename of model output files")
    ("log-period", value<unsigned int>()->default_value(0),
        "Log model every X iterations")
    ("lambda,r", value<float>()->default_value(7.0),
        "regularisation strength parameter")
    ("dump-frequency", value<int>()->default_value(0),
        "dump model every n minibatches.")
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("test-tokens", value<int>()->default_value(10000),
        "number of evenly spaced test points tokens evaluate.")
    ("step-size", value<float>()->default_value(0.05),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("classes", value<int>()->default_value(100),
        "number of classes for factored output.")
    ("class-file", value<string>(),
        "file containing word to class mappings in the format <class> <word> <frequence>.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
    ("randomise", "visit the training tokens in random order")
    ("reclass", "reallocate word classes after the first epoch.")
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
  config.feature_context_size = vm["feature-context-size"].as<unsigned int>();
  config.verbose = vm.count("verbose");
  config.classes = vm["classes"].as<int>();

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  cerr << "# order = " << vm["order"].as<int>() << endl;
  cerr << "# feature context size = " << config.feature_context_size << endl;
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


void learn(const variables_map& vm, ModelData& config) {
  Corpus training_corpus, test_corpus;
  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");

  //////////////////////////////////////////////
  // separate the word types into classes using
  // frequency binning
  vector<int> classes;
  VectorReal class_bias = VectorReal::Zero(vm["classes"].as<int>());
  if (vm.count("class-file")) {
    cerr << "--class-file set, ignoring --classes." << endl;
    classes_from_file(vm["class-file"].as<string>(), classes, dict, class_bias);
    config.classes = classes.size()-1;
  }
  else
    freq_bin_type(vm["input"].as<string>(), vm["classes"].as<int>(), classes, dict, class_bias);
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

  FactoredMaxentNLM model(config, dict, vm.count("diagonal-contexts"), classes);
  model.FB = class_bias;

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
  VectorReal global_gradientFB(model.FB.size());
  MatrixReal adaGradF = MatrixReal::Zero(model.F.rows(), model.F.cols());
  VectorReal adaGradFB = VectorReal::Zero(model.FB.size());

  UnconstrainedFeatureStore global_gradientU(config.classes);
  UnconstrainedFeatureStore adaGradU(config.classes);
  vector<UnconstrainedFeatureStore> global_gradientV, adaGradV;
  for (int i = 0; i < config.classes; ++i) {
    int num_words_in_class = classes[i + 1] - classes[i];
    global_gradientV.push_back(UnconstrainedFeatureStore(num_words_in_class));
    adaGradV.push_back(UnconstrainedFeatureStore(num_words_in_class));
  }

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
    NLM::WeightsType gradient(gradient_data, model.num_weights());

    NLM::WordVectorsType g_R(gradient_data, num_words, word_width);
    NLM::WordVectorsType g_Q(gradient_data+R_size, num_words, word_width);

    NLM::ContextTransformsType g_C;
    Real* ptr = gradient_data+2*R_size;
    for (int i=0; i<context_width; i++) {
      if (vm.count("diagonal-contexts"))
          g_C.push_back(NLM::ContextTransformType(ptr, word_width, 1));
      else
          g_C.push_back(NLM::ContextTransformType(ptr, word_width, word_width));
      ptr += C_size;
    }

    NLM::WeightsType g_B(ptr, B_size);
    NLM::WeightsType g_M(ptr+B_size, M_size);
    MatrixReal g_F(num_classes, word_width);
    VectorReal g_FB(num_classes);
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
          global_gradientFB.setZero();
          global_gradientU.clear();
          for (int i = 0; i < num_classes; ++i) {
            global_gradientV[i].clear();
          }
        }

        gradient.setZero();
        g_F.setZero();
        g_FB.setZero();
        UnconstrainedFeatureStore g_U(num_classes);
        vector<UnconstrainedFeatureStore> g_V;
        for (int i = 0; i < num_classes; ++i) {
          g_V.push_back(UnconstrainedFeatureStore(classes[i + 1] - classes[i]));
        }
        Real lambda = config.l2_parameter*(end-start)/static_cast<Real>(training_corpus.size());

        #pragma omp barrier
        cache_data(start, end, training_corpus, training_indices, training_instances);
        Real f = sgd_gradient(model, training_corpus, training_instances,
                              lambda, g_R, g_Q, g_C, g_B, g_F, g_FB, g_U, g_V);

        #pragma omp critical
        {
          global_gradient += gradient;
          global_gradientF += g_F;
          global_gradientFB += g_FB;
          global_gradientU.update(g_U);
          for (int i = 0; i < num_classes; ++i) {
            global_gradientV[i].update(g_V[i]);
          }
          av_f += f;
        }
        #pragma omp barrier
        #pragma omp master
        {
          adaGrad.array() += global_gradient.array().square();
          for (int w=0; w<model.num_weights(); ++w)
            if (adaGrad(w)) model.W(w) -= (step_size*global_gradient(w) / sqrt(adaGrad(w)));

          adaGradF.array() += global_gradientF.array().square();
          adaGradFB.array() += global_gradientFB.array().square();
          for (int r=0; r < adaGradF.rows(); ++r) {
            if (adaGradFB(r)) model.FB(r) -= (step_size*global_gradientFB(r) / sqrt(adaGradFB(r)));
            for (int c=0; c < adaGradF.cols(); ++c)
              if (adaGradF(r,c)) model.F(r,c) -= (step_size*global_gradientF(r,c) / sqrt(adaGradF(r,c)));
          }

          adaGradU.updateSquared(global_gradientU);
          for (int i = 0; i < num_classes; ++i) {
            adaGradV[i].updateSquared(global_gradientV[i]);
          }

          model.U.updateAdaGrad(global_gradientU, adaGradU, step_size);
          for (int i = 0; i < num_classes; ++i) {
            model.V[i].updateAdaGrad(global_gradientV[i], adaGradV[i], step_size);
          }

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
        Real local_pp = perplexity(model, test_corpus);

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
        cerr << " |" << endl << endl;

        if (iteration >= 1 && vm.count("reclass")) {
          model.reclass(training_corpus, test_corpus);
          adaGradF = MatrixReal::Zero(model.F.rows(), model.F.cols());
          adaGradFB = VectorReal::Zero(model.FB.size());
          adaGrad = VectorReal::Zero(model.num_weights());
        }
      }

      if (vm.count("model-out") && vm.count("log-period")) {
        unsigned int log_period = vm["log-period"].as<unsigned int>();
        if (log_period > 0 && iteration % log_period == 0) {
          string file = vm["model-out"].as<string>() + ".i" + to_string(iteration);
          cout << "Writing trained model to " << file << endl;
          std::ofstream f(file);
          boost::archive::text_oarchive ar(f);
          ar << model;
        }
      }
    }
  }

  string file = vm["model-out"].as<string>();
  cout << "Writing final trained model to " << file << endl;
  std::ofstream f(file);
  boost::archive::text_oarchive ar(f);
  ar << model;
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


Real sgd_gradient(FactoredMaxentNLM& model,
                const Corpus& training_corpus,
                const TrainingInstances &training_instances,
                Real lambda,
                NLM::WordVectorsType& g_R,
                NLM::WordVectorsType& g_Q,
                NLM::ContextTransformsType& g_C,
                NLM::WeightsType& g_B,
                MatrixReal& g_F,
                VectorReal& g_FB,
                UnconstrainedFeatureStore& g_U,
                vector<UnconstrainedFeatureStore>& g_V) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");
  WordId end_id = model.label_set().Convert("</s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // form matrices of the ngram histories
//  clock_t cache_start = clock();
  int instances=training_instances.size();
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width));
  vector<vector<int>> histories(instances, vector<int>(context_width));
  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    int context_start = t - context_width;

    bool sentence_start = (t==0);
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || training_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : training_corpus.at(j));
      histories[instance][i] = v_i;
      context_vectors.at(i).row(instance) = model.Q.row(v_i);
    }
  }

  MatrixReal prediction_vectors = MatrixReal::Zero(instances, word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += model.context_product(i, context_vectors.at(i));

//  clock_t cache_time = clock() - cache_start;

  // the weighted sum of word representations
  MatrixReal weightedRepresentations = MatrixReal::Zero(instances, word_width);

  // calculate the function and gradient for each ngram
//  clock_t iteration_start = clock();
  FeatureGenerator feature_generator(model.config.feature_context_size);
  for (int instance=0; instance < instances; instance++) {
    int w_i = training_instances.at(instance);
    WordId w = training_corpus.at(w_i);
    int c = model.get_class(w);
    int c_start = model.indexes.at(c), c_end = model.indexes.at(c+1);

    vector<Feature> features = feature_generator.generate(histories[instance]);

    if (!(w >= c_start && w < c_end))
      cerr << w << " " << c << " " << c_start << " " << c_end << endl;
    assert(w >= c_start && w < c_end);

    // a simple sigmoid non-linearity
    prediction_vectors.row(instance) = sigmoid(prediction_vectors.row(instance));
    //for (int x=0; x<word_width; ++x)
    //  prediction_vectors.row(instance)(x) *= (prediction_vectors.row(instance)(x) > 0 ? 1 : 0.01); // rectifier

    VectorReal class_feature_scores = model.U.get(features);
    VectorReal word_feature_scores = model.V[c].get(features);
    VectorReal class_conditional_scores = model.F * prediction_vectors.row(instance).transpose() + model.FB + class_feature_scores;
    VectorReal word_conditional_scores  = model.class_R(c) * prediction_vectors.row(instance).transpose() + model.class_B(c) + word_feature_scores;

    ArrayReal class_conditional_log_probs = logSoftMax(class_conditional_scores);
    ArrayReal word_conditional_log_probs  = logSoftMax(word_conditional_scores);

    VectorReal class_conditional_probs = class_conditional_log_probs.exp();
    VectorReal word_conditional_probs  = word_conditional_log_probs.exp();

    weightedRepresentations.row(instance) -= (model.F.row(c) - class_conditional_probs.transpose() * model.F);
    weightedRepresentations.row(instance) -= (model.R.row(w) - word_conditional_probs.transpose() * model.class_R(c));

    assert(isfinite(class_conditional_log_probs(c)));
    assert(isfinite(word_conditional_log_probs(w-c_start)));
    f -= (class_conditional_log_probs(c) + word_conditional_log_probs(w-c_start));

    // do the gradient updates:
    class_conditional_probs(c) -= 1;
    word_conditional_probs(w-c_start) -= 1;

    g_F += class_conditional_probs * prediction_vectors.row(instance);
    g_FB += class_conditional_probs;
    g_R.block(c_start, 0, c_end-c_start, g_R.cols()) += word_conditional_probs * prediction_vectors.row(instance);
    g_B.segment(c_start, c_end-c_start) += word_conditional_probs;
    g_U.update(features, class_conditional_probs);
    g_V[c].update(features, word_conditional_probs);

    // a simple sigmoid non-linearity derivative
    weightedRepresentations.row(instance).array() *=
      prediction_vectors.row(instance).array() * (1.0 - prediction_vectors.row(instance).array()); // sigmoid
    //for (int x=0; x<word_width; ++x)
    //  weightedRepresentations.row(instance)(x) *= (prediction_vectors.row(instance)(x) > 0 ? 1 : 0.01); // rectifier
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

Real perplexity(const FactoredMaxentNLM& model, const Corpus& test_corpus) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;
  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

  #pragma omp master
  cerr << "Calculating perplexity for " << test_corpus.size() << " tokens";

  std::vector<WordId> context(context_width);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t s = thread_num; s < test_corpus.size(); s += num_threads) {
    WordId w = test_corpus.at(s);
    int context_start = s - context_width;
    bool sentence_start = (s==0);

    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || test_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : test_corpus.at(j));

      context.at(i) = v_i;
    }
    Real log_prob = model.log_prob(w, context, true, false);
    p += log_prob;

    #pragma omp master
    if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }

    tokens++;
  }
  #pragma omp master
  cerr << endl;

  return p;
}


void freq_bin_type(const std::string &corpus, int num_classes, vector<int>& classes, Dict& dict, VectorReal& class_bias) {
  ifstream in(corpus.c_str());
  string line, token;

  map<string,int> tmp_dict;
  vector< pair<string,int> > counts;
  int sum=0, eos_sum=0;
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
    eos_sum++;
  }

  sort(counts.begin(), counts.end(),
       [](const pair<string,int>& a, const pair<string,int>& b) -> bool { return a.second > b.second; });

  classes.clear();
  classes.push_back(0);

  classes.push_back(2);
  class_bias(0) = log(eos_sum);
  int bin_size = sum / (num_classes-1);

//  int bin_size = counts.size()/(num_classes);

  int mass=0;
  for (int i=0; i < int(counts.size()); ++i) {
    WordId id = dict.Convert(counts.at(i).first);

//    if ((mass += 1) > bin_size) {

    if ((mass += counts.at(i).second) > bin_size) {
      bin_size = (sum -= mass) / (num_classes - classes.size());
      class_bias(classes.size()-1) = log(mass);


//      class_bias(classes.size()-1) = 1;

      classes.push_back(id+1);

//      cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
      mass=0;
    }
  }
  if (classes.back() != int(dict.size()))
    classes.push_back(dict.size());

//  cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
  class_bias.array() -= log(eos_sum+sum);

  cerr << "Binned " << dict.size() << " types in " << classes.size()-1 << " classes with an average of "
       << float(dict.size()) / float(classes.size()-1) << " types per bin." << endl;
  in.close();
}

void classes_from_file(const std::string &class_file, vector<int>& classes, Dict& dict, VectorReal& class_bias) {
  ifstream in(class_file.c_str());

  vector<int> class_freqs(1,0);
  classes.clear();
  classes.push_back(0);
  classes.push_back(2);

  int mass=0, total_mass=0;
  string prev_class_str="", class_str="", token_str="", freq_str="";
  while (in >> class_str >> token_str >> freq_str) {
    int w_id = dict.Convert(token_str);

    if (!prev_class_str.empty() && class_str != prev_class_str) {
      class_freqs.push_back(log(mass));
      classes.push_back(w_id);
//      cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
      mass=0;
    }

    int freq = lexical_cast<int>(freq_str);
    mass += freq;
    total_mass += freq;

    prev_class_str=class_str;
  }

  class_freqs.push_back(log(mass));
  classes.push_back(dict.size());
//  cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;

  class_bias = VectorReal::Zero(class_freqs.size());
  for (size_t i=0; i<class_freqs.size(); ++i)
    class_bias(i) = class_freqs.at(i) - log(total_mass);

  cerr << "Read " << dict.size() << " types in " << classes.size()-1 << " classes with an average of "
       << float(dict.size()) / float(classes.size()-1) << " types per bin." << endl;

  in.close();
}
