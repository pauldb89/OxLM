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

struct TrainingInstance {
  vector<WordId> noise_words; 
  Real            data_prob;
  int             data_index;
  vector<Real>    noise_probs;
};
typedef vector<TrainingInstance> TrainingInstances;
void cache_data(const LogBiLinearModel& model,
                int start, int end, 
                const Corpus& training_corpus, const vector<size_t>& indices, const VectorReal& unigram,
                int k, TrainingInstances &result);

Real sgd_update(LogBiLinearModel& model, int start, int end, 
                const Corpus& training_corpus, 
                const TrainingInstances &result,
                Real lambda, Real step_size);
Real perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride=1);


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
        "number of evenly spaced test points tokens evaluate.")
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

  vector<size_t> training_indices(training_corpus.size());
  VectorReal unigram = VectorReal::Zero(model.labels());
  for (size_t i=0; i<training_indices.size(); i++) {
    unigram(training_corpus[i]) += 1;
    training_indices[i] = i;
  }
  model.B = ((unigram.array()+1.0)/(unigram.sum()+unigram.size())).log();
  unigram /= unigram.sum();

  size_t minibatch_counter=0;
  size_t minibatch_size = vm["minibatch-size"].as<int>();
  for (int iteration=0; iteration < vm["iterations"].as<int>(); ++iteration) {
    clock_t iteration_start=clock();
    Real av_f=0.0;
    cout << "Iteration " << iteration << ": "; cout.flush();

    if (vm.count("randomise"))
      std::random_shuffle(training_indices.begin(), training_indices.end());

    #pragma omp parallel \
      firstprivate(minibatch_counter) \
      shared(training_corpus,training_indices,model,vm,minibatch_size,iteration,config,unigram) \
      reduction(+:av_f)
    {
      TrainingInstances training_instances;

      int thread_id = omp_get_thread_num();
      for (size_t start=thread_id*minibatch_size; start < training_corpus.size(); ++minibatch_counter) {
        size_t end = min(training_corpus.size(), start + minibatch_size);

        Real lambda = config.l2_parameter*(end-start)/static_cast<Real>(training_corpus.size()); 

        cache_data(model, start, end, training_corpus, training_indices, unigram,
                   config.label_sample_size, training_instances);

        Real f=0.0;
        Real step_size = vm["step-size"].as<float>(); //* minibatch_size / training_corpus.size();
        f = sgd_update(model, start, end, training_corpus, training_instances, lambda, step_size);
        av_f += f;

        // regularisation
        if (lambda > 0) model.l2_gradient_update(step_size*lambda);

        #pragma omp master 
        if (minibatch_counter % 100 == 0) { cerr << "."; cout.flush(); }

        start += (minibatch_size*omp_get_num_threads());
      }
    }

    Real iteration_time = (clock()-iteration_start) / (Real)CLOCKS_PER_SEC;
    cerr << "\n | Time: " << iteration_time << " seconds, Average f = " << av_f/training_corpus.size();
    if (vm.count("test-set"))
      cerr << ", Test Perplexity = " << perplexity(model, test_corpus, test_corpus.size()/vm["test-tokens"].as<int>()); 
    cerr << " |" << endl << endl;
  }

  if (vm.count("model-out")) {
    cout << "Writing trained model to " << vm["model-out"].as<string>() << endl;
    std::ofstream f(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f);
    ar << model;
  }
}


void cache_data(const LogBiLinearModel& model,
                int start, int end, const Corpus& training_corpus, const vector<size_t>& indices, 
                const VectorReal& unigram, int k, TrainingInstances &result) {
  assert (start>=0 && start < end && end <= static_cast<int>(training_corpus.size()));
  assert (training_corpus.size() == indices.size());
  int num_tokens = training_corpus.size();
  int num_words = model.labels();

  result.resize(end-start);
  for (int s=start; s<end; ++s) {
    TrainingInstance& t = result.at(s-start);
    int w_i = indices.at(s);
    t.data_index = w_i;
    WordId w = training_corpus.at(w_i);

    t.noise_words.resize(k);
    t.noise_probs.resize(k);
    if (model.config.uniform) {
      Real word_prob = 1.0 / model.labels();
      t.data_prob = word_prob;
      for(int n_i=0; n_i < k; n_i++) {
        t.noise_words.at(n_i) = rand() % num_words;
        t.noise_probs.at(n_i) = word_prob;
      }
    }
    else {
      t.data_prob = unigram(w);
      for(int n_i=0; n_i < k; ++n_i) {
        WordId n_w = training_corpus.at(rand() % num_tokens);
        t.noise_words.at(n_i) = n_w;
        t.noise_probs.at(n_i) = unigram(n_w);
      }
    }
  }
}


Real sgd_update(LogBiLinearModel& model,
                int start, int end, const Corpus& training_corpus,
                const TrainingInstances &training_instances,
                Real lambda, Real step_size) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // form matrices of the ngram histories
//  clock_t cache_start = clock();
  int instances=end-start;
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width)); 
  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    int context_start = t.data_index - context_width;

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
  Real unnormalised_llh=0;

//  clock_t iteration_start = clock();
  vector<Real> pos_probs(instances);
  vector<Real> neg_probs; 
  neg_probs.reserve(instances*(training_instances.empty() ? 0 : training_instances.front().noise_words.size()));
  
  for (int instance=0; instance < instances; instance++) {
    const TrainingInstance& t = training_instances.at(instance);
    WordId w = training_corpus.at(t.data_index);

    // data update
    Real k = t.noise_words.size();
    Real log_posP = (model.R.row(w) * prediction_vectors.row(instance).transpose()) + model.B(w);
    Real posP = exp(log_posP);
    posP = posP > numeric_limits<Real>::max() ? numeric_limits<Real>::max() : posP;
    Real posW = k*t.data_prob / (posP + k*t.data_prob);

    Real log_pos_nc_p = (log_posP - Log<Real>::add(log_posP, log(k*t.data_prob)));
    log_pos_nc_p = log_pos_nc_p < -numeric_limits<Real>::max() ? -numeric_limits<Real>::max() : log_pos_nc_p;
    assert(abs(log_pos_nc_p) <= numeric_limits<Real>::max());
    f -= log_pos_nc_p;

    unnormalised_llh += log_posP;

    weightedRepresentations.row(instance) -= posW * model.R.row(w).transpose();
    pos_probs.at(instance) = posW;

    for (size_t nl_i=0; nl_i < t.noise_words.size(); ++nl_i) {
      WordId v_noise = t.noise_words.at(nl_i);
      Real log_negP = model.R.row(v_noise) * prediction_vectors.row(instance).transpose() + model.B(v_noise);
      Real negP = exp(log_negP);
      negP = negP > numeric_limits<Real>::max() ? numeric_limits<Real>::max() : negP;
      Real negW = negP / (negP + k*t.noise_probs.at(nl_i));

      Real log_negNoise_nc_p = log(k*t.noise_probs.at(nl_i)) - Log<Real>::add(log_negP, log(k*t.noise_probs.at(nl_i)));
      log_negNoise_nc_p = log_negNoise_nc_p < -numeric_limits<Real>::max() ? -numeric_limits<Real>::max() : log_negNoise_nc_p;
      f -= log_negNoise_nc_p;

      weightedRepresentations.row(instance) += negW * model.R.row(v_noise).transpose();
      neg_probs.push_back(negW);
      nl_i++;
    }
  }

  // do the gradient updates
  int noise_prob_index=0;
  for (int instance=0; instance < instances; instance++) {
    const TrainingInstance& t = training_instances.at(instance);
    WordId w = training_corpus.at(t.data_index);
    Real posW = pos_probs.at(instance);

    model.R.row(w) += step_size * posW * prediction_vectors.row(instance).transpose();
    model.B(w) += step_size * posW;

    for (size_t nl_i=0; nl_i < t.noise_words.size(); ++nl_i, ++noise_prob_index) {
      WordId v_noise = t.noise_words.at(nl_i);
      Real negW = neg_probs.at(noise_prob_index);

      model.R.row(v_noise) -= step_size * negW * prediction_vectors.row(instance);
      model.B(v_noise) -= step_size * negW;

      nl_i++;
    }
  }
//  clock_t iteration_time = clock() - iteration_start;

//  clock_t context_start = clock();
  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i=0; i<context_width; ++i) {
    context_gradients = (model.C.at(i) * weightedRepresentations.transpose()).transpose();
    for (int instance=0; instance < instances; ++instance) {
      const TrainingInstance& t = training_instances.at(instance);
      int j=t.data_index-context_width+i;
      int v_i = (j<0 ? start_id : training_corpus.at(j));
      model.Q.row(v_i) -= step_size * context_gradients.row(instance);
    }
    model.C.at(i) -= step_size * context_vectors.at(i).transpose() * weightedRepresentations; 
  }
//  clock_t context_time = clock() - context_start;

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
