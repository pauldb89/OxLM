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

void learn(const ModelData& config);

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

Real sgd_gradient(LogBiLinearModel& model,
                const Corpus& training_corpus,
                const TrainingInstances &result,
                bool pseudo,
                LogBiLinearModel::WordVectorsType& g_R,
                LogBiLinearModel::WordVectorsType& g_Q,
                LogBiLinearModel::ContextTransformsType& g_C,
                LogBiLinearModel::WeightsType& g_B);

Real mixture_sgd_gradient(LogBiLinearModel& model,
                          const Corpus& training_corpus,
                          const TrainingInstances &result,
                          LogBiLinearModel::WordVectorsType& g_R,
                          LogBiLinearModel::WordVectorsType& g_Q,
                          LogBiLinearModel::ContextTransformsType& g_C,
                          LogBiLinearModel::WeightsType& g_B,
                          LogBiLinearModel::WeightsType& g_M);

Real perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride=1);
Real mixture_perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride=1);


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
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("step-size", value<float>()->default_value(1.0),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("randomise", "visit the training tokens in random order")
    ("uniform", "sample noise distribution from a uniform (default unigram) distribution.")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ("pseudo", "Use a pseudo-likelihood CNE objective.")
    ("mixture", "Train a mixture of bigram LBL models, one per context position.");
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
  config.instances = vm["instances"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  if (vm.count("model-in")) {
    config.model_input_file = vm["model-in"].as<string>();
  }
  if (vm.count("model-out")) {
    config.model_output_file = vm["model-out"].as<string>();
  }
  config.l2_lbl = vm["lambda-lbl"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.step_size = vm["step-size"].as<float>();
  config.randomise = vm.count("randomise");
  config.uniform = vm.count("uniform");
  config.diagonal_contexts = vm.count("diagonal-contexts");
  config.pseudo_likelihood_cne = vm.count("pseudo");
  config.mixture = vm.count("mixture");

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
  cerr << "# lambda = " << config.l2_lbl << endl;
  cerr << "# iterations = " << config.iterations << endl;
  cerr << "# threads = " << config.threads << endl;
  cerr << "# mixture = " << config.threads << endl;
  cerr << "# pseudo = " << config.pseudo << endl;
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

  LogBiLinearModel model(config, dict);

  if (config.model_input_file.size()) {
    std::ifstream f(config.model_input_file);
    boost::archive::text_iarchive ar(f);
    ar >> model;
  }

  vector<size_t> training_indices(training_corpus.size());
  //VectorReal unigram = VectorReal::Zero(model.labels());
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

  #pragma omp parallel shared(global_gradient)
  {
    //////////////////////////////////////////////
    // setup the gradient matrices
    int num_words = model.labels();
    int word_width = model.config.word_representation_size;
    int context_width = model.config.ngram_order-1;

    int R_size = num_words*word_width;
    int Q_size = R_size;
    int C_size = config.diagonal_contexts ? word_width : word_width*word_width;
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
      if (config.diagonal_contexts) {
        g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, 1));
      } else {
        g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, word_width));
      }
      ptr += C_size;
    }

    LogBiLinearModel::WeightsType g_B(ptr, B_size);
    LogBiLinearModel::WeightsType g_M(ptr+B_size, M_size);
    //////////////////////////////////////////////

    size_t minibatch_counter=0;
    size_t minibatch_size = config.minibatch_size;
    for (int iteration=0; iteration < config.iterations; ++iteration) {
      clock_t iteration_start=clock();
      #pragma omp master
      {
        av_f=0.0;
        pp=0.0;
        cout << "Iteration " << iteration << ": "; cout.flush();

        if (config.randomise)
          std::random_shuffle(training_indices.begin(), training_indices.end());
      }

      TrainingInstances training_instances;
//      int thread_id = omp_get_thread_num();
      Real step_size = config.step_size;

      //for (size_t start=thread_id*minibatch_size; start < training_corpus.size(); ++minibatch_counter) {
      for (size_t start=0; start < training_corpus.size() && (int)start < config.instances; ++minibatch_counter) {
        size_t end = min(training_corpus.size(), start + minibatch_size);

        #pragma omp master
        global_gradient.setZero();

        gradient.setZero();

        #pragma omp barrier
        cache_data(model, start, end, training_corpus, training_indices, model.unigram,
                   config.label_sample_size, training_instances);

        Real f=0.0;
        if (config.mixture)
          f = mixture_sgd_gradient(model, training_corpus, training_instances,
                                   g_R, g_Q, g_C, g_B, g_M);
        else
          f = sgd_gradient(model, training_corpus, training_instances,
                           config.pseudo_likelihood_cne, g_R, g_Q, g_C, g_B);

        #pragma omp critical
        {
          global_gradient += gradient;
          av_f += f;
        }
        #pragma omp barrier
        #pragma omp master
        {
          adaGrad.array() += global_gradient.array().square();
          for (int w=0; w<model.num_weights(); ++w)
            if (adaGrad(w)) model.W(w) -= (step_size*global_gradient(w) / sqrt(adaGrad(w)));

        //        model.W -= (step_size*global_gradient);
        //        model.R -= (step_size*g_R);
        //        model.Q -= (step_size*g_Q);
        //        for (int i=0; i<context_width; i++)
        //          model.C.at(i) -= (step_size*g_C.at(i));
        //        model.B -= (step_size*g_B);

          // regularisation
          if (model.l2_lbl > 0) {
            Real minibatch_factor = static_cast<Real>(end - start) / training_corpus.size();
            model.l2GradientUpdate(minibatch_factor);
          }

          if (minibatch_counter % 100 == 0) { cerr << "."; cout.flush(); }
        }

        //start += (minibatch_size*omp_get_num_threads());
        start += minibatch_size;
      }
      #pragma omp master
      cerr << endl;

      Real iteration_time = (clock()-iteration_start) / (Real)CLOCKS_PER_SEC;
      if (test_corpus.size()) {
        Real local_pp=0;
        if (config.mixture)
          local_pp = mixture_perplexity(model, test_corpus, 1);
          //local_pp = mixture_perplexity(model, test_corpus, max(1,(int)test_corpus.size()/test_tokens));
        else
          local_pp = perplexity(model, test_corpus, 1);
          //local_pp = perplexity(model, test_corpus, max(1,(int)test_corpus.size()/test_tokens));

        #pragma omp critical
        { pp += local_pp; }
        #pragma omp barrier
      }

      #pragma omp master
      {
        pp = exp(-pp/test_corpus.size());
        cerr << " | Time: " << iteration_time << " seconds, Average f = " << av_f/training_corpus.size();
        if (test_corpus.size()) {
          cerr << ", Test Perplexity = " << pp;
        }
        if (config.mixture)
          cerr << ", Mixture weights = " << softMax(model.M).transpose();
        cerr << " |" << endl << endl;
      }
    }
  }

  if (config.model_output_file.size()) {
    cout << "Writing trained model to " << config.model_output_file << endl;
    std::ofstream f(config.model_output_file);
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

  result.clear();
  result.reserve(end-start);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  for (int s = start+thread_num; s < end; s += num_threads) {
    result.push_back(TrainingInstance());
    TrainingInstance& t = result.back();

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


Real sgd_gradient(LogBiLinearModel& model,
                const Corpus& training_corpus,
                const TrainingInstances &training_instances,
                bool pseudo,
                LogBiLinearModel::WordVectorsType& g_R,
                LogBiLinearModel::WordVectorsType& g_Q,
                LogBiLinearModel::ContextTransformsType& g_C,
                LogBiLinearModel::WeightsType& g_B) {
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
    int context_start = t.data_index - context_width;

    bool sentence_start = (t.data_index==0);
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


  vector<MatrixReal> rev_context_vectors;
  MatrixReal rev_prediction_vectors;
  if (pseudo) {
    rev_context_vectors.resize(context_width, MatrixReal::Zero(instances, word_width));
    for (int instance=0; instance < instances; ++instance) {
      const TrainingInstance& t = training_instances.at(instance);
      int context_end = t.data_index + context_width;

      bool sentence_end = (t.data_index== int(training_corpus.size())-1);
      for (int i=context_width-1; i>=0 && !sentence_end; --i) {
        int j=context_end-i;
        assert (j < int(training_corpus.size()));
        int v_i = training_corpus.at(j);
        sentence_end = (v_i == end_id);
        rev_context_vectors.at(i).row(instance) = model.R.row(v_i);
      }
    }
    rev_prediction_vectors = MatrixReal::Zero(instances, word_width);
    for (int i=0; i<context_width; ++i)
      rev_prediction_vectors += model.context_product(i, rev_context_vectors.at(i), true);
  }

  // Drop out masking of the prediction vectors
  //cerr << "HERE" << endl;
  //ArrayReal drop_out = Eigen::ArrayXf::Random(instances, word_width);
  //cerr << "THERE" << endl;
  /*
  MatrixReal drop_out = MatrixReal::Ones(instances, word_width);
  if (rand()%2)
    for (int i=0; i<drop_out.rows(); ++i)
      for (int j=0; j<drop_out.cols(); ++j)
        drop_out(i,j) = (rand()%2==0 ? 1.0 : 0.0);

  //ArrayReal drop_out = (MatrixReal::Random(instances, word_width) > 0.0f).cast<Real>();
  prediction_vectors.array() = prediction_vectors.array()*drop_out.array();
  */



//  clock_t cache_time = clock() - cache_start;

  // calculate the weight sum of word representations
  MatrixReal weightedRepresentations = MatrixReal::Zero(instances, word_width);
  MatrixReal rev_weightedRepresentations;
  if (pseudo) { rev_weightedRepresentations = MatrixReal::Zero(instances, word_width); }

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
    if (pseudo) log_posP += model.Q.row(w) * rev_prediction_vectors.row(instance).transpose();

    Real posP = exp(log_posP);
    posP = posP > numeric_limits<Real>::max() ? numeric_limits<Real>::max() : posP;
    Real posW = k*t.data_prob / (posP + k*t.data_prob);

    Real log_pos_nc_p = (log_posP - Log<Real>::add(log_posP, log(k*t.data_prob)));
    log_pos_nc_p = log_pos_nc_p < -numeric_limits<Real>::max() ? -numeric_limits<Real>::max() : log_pos_nc_p;
    assert(abs(log_pos_nc_p) <= numeric_limits<Real>::max());
    f -= log_pos_nc_p;

    unnormalised_llh += log_posP;

    weightedRepresentations.row(instance) -= posW * model.R.row(w).transpose();
    if (pseudo) rev_weightedRepresentations.row(instance) -= posW * model.Q.row(w).transpose();
    pos_probs.at(instance) = posW;

    for (size_t nl_i=0; nl_i < t.noise_words.size(); ++nl_i) {
      WordId v_noise = t.noise_words.at(nl_i);
      Real log_negP = model.R.row(v_noise) * prediction_vectors.row(instance).transpose() + model.B(v_noise);
      if (pseudo) log_negP += model.Q.row(v_noise) * rev_prediction_vectors.row(instance).transpose();

      Real negP = exp(log_negP);
      negP = negP > numeric_limits<Real>::max() ? numeric_limits<Real>::max() : negP;
      Real negW = negP / (negP + k*t.noise_probs.at(nl_i));

      Real log_negNoise_nc_p = log(k*t.noise_probs.at(nl_i)) - Log<Real>::add(log_negP, log(k*t.noise_probs.at(nl_i)));
      log_negNoise_nc_p = log_negNoise_nc_p < -numeric_limits<Real>::max() ? -numeric_limits<Real>::max() : log_negNoise_nc_p;
      f -= log_negNoise_nc_p;

      weightedRepresentations.row(instance) += negW * model.R.row(v_noise).transpose();
      if (pseudo) rev_weightedRepresentations.row(instance) += negW * model.Q.row(v_noise).transpose();
      neg_probs.push_back(negW);
    }
  }

  // do the gradient updates
  int noise_prob_index=0;
  for (int instance=0; instance < instances; instance++) {
    const TrainingInstance& t = training_instances.at(instance);
    WordId w = training_corpus.at(t.data_index);
    Real posW = pos_probs.at(instance);

    //model.R.row(w) += step_size * posW * prediction_vectors.row(instance).transpose();
    //model.B(w) += step_size * posW;
    g_R.row(w) -= posW * prediction_vectors.row(instance).transpose();
    if (pseudo) g_Q.row(w) -= posW * rev_prediction_vectors.row(instance).transpose();
    g_B(w) -= posW;

    for (size_t nl_i=0; nl_i < t.noise_words.size(); ++nl_i, ++noise_prob_index) {
      WordId v_noise = t.noise_words.at(nl_i);
      Real negW = neg_probs.at(noise_prob_index);

      //model.R.row(v_noise) -= step_size * negW * prediction_vectors.row(instance);
      //model.B(v_noise) -= step_size * negW;
      g_R.row(v_noise) += negW * prediction_vectors.row(instance);
      if (pseudo) g_Q.row(v_noise) += negW * rev_prediction_vectors.row(instance);
      g_B(v_noise) += negW;
    }
  }
//  clock_t iteration_time = clock() - iteration_start;
  //weightedRepresentations.array() = weightedRepresentations.array()*drop_out.array();

//  clock_t context_start = clock();
  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  MatrixReal rev_context_gradients;
  if (pseudo) rev_context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i=0; i<context_width; ++i) {
    context_gradients = model.context_product(i, weightedRepresentations, true); // weightedRepresentations*C(i)^T
    if (pseudo) rev_context_gradients = model.context_product(i, rev_weightedRepresentations); // rev_weightedRepresentations*C(i)

    for (int instance=0; instance < instances; ++instance) {
      const TrainingInstance& t = training_instances.at(instance);
      int j=t.data_index-context_width+i;

      bool sentence_start = (j<0);
      for (int k=j; !sentence_start && k < t.data_index; k++)
        if (training_corpus.at(k) == end_id)
          sentence_start=true;
      int v_i = (sentence_start ? start_id : training_corpus.at(j));

      //model.Q.row(v_i) -= step_size * context_gradients.row(instance);
      g_Q.row(v_i) += context_gradients.row(instance);

      if (pseudo) {
        int k=t.data_index+context_width-i;
        if (k<instances) {
          int w_i = training_corpus.at(k);
          g_R.row(w_i) += rev_context_gradients.row(instance);
        }
      }
    }
    //model.C.at(i) -= step_size * context_vectors.at(i).transpose() * weightedRepresentations;
    model.context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations);
    if (pseudo)
      model.context_gradient_update(g_C.at(i), rev_weightedRepresentations, rev_context_vectors.at(i));
  }
//  clock_t context_time = clock() - context_start;

  return f;
}


Real mixture_sgd_gradient(LogBiLinearModel& model,
                          const Corpus& training_corpus,
                          const TrainingInstances &training_instances,
                          LogBiLinearModel::WordVectorsType& g_R,
                          LogBiLinearModel::WordVectorsType& g_Q,
                          LogBiLinearModel::ContextTransformsType& g_C,
                          LogBiLinearModel::WeightsType& g_B,
                          LogBiLinearModel::WeightsType& g_M) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");
  WordId end_id = model.label_set().Convert("</s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // form matrices of the ngram histories
  int instances=training_instances.size();
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width));

  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    int context_start = t.data_index - context_width;

    bool sentence_start = (t.data_index==0);
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || training_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : training_corpus.at(j));
      context_vectors.at(i).row(instance) = model.Q.row(v_i);
    }
  }
  vector<MatrixReal> prediction_vectors(context_width, MatrixReal::Zero(instances, word_width));

  for (int i=0; i<context_width; ++i)
    prediction_vectors.at(i) = model.context_product(i, context_vectors.at(i));

  // the weight sum of word representations
  vector<MatrixReal> weightedRepresentations(context_width, MatrixReal::Zero(instances, word_width));

  // calculate the function and gradient for each ngram
  Real unnormalised_llh=0;

  // cache the mixture weights
  VectorReal pM = softMax(model.M);


  for (int instance=0; instance < instances; instance++) {
    const TrainingInstance& t = training_instances.at(instance);
    WordId w = training_corpus.at(t.data_index);

    // calculate the log-probabilities
    VectorReal logProbs(context_width);
    for (int i=0; i<context_width; i++)
      logProbs(i) = model.R.row(w) * prediction_vectors.at(i).row(instance).transpose() + model.B(w);

    // calculate the mixture contributions
    VectorReal mixLogProbs = logProbs.array() + log(pM.array());
    Real max = mixLogProbs.maxCoeff();
    Real contributions_log_z = log((mixLogProbs.array() - max).exp().sum()) + max;
    VectorReal contributions = (mixLogProbs.array() - contributions_log_z).exp();

    // data update
    Real k = t.noise_words.size();
    //Real log_posP = (model.R.row(w) * prediction_vectors.row(instance).transpose()) + model.B(w);
    Real log_posP = contributions_log_z;
    Real posP = exp(log_posP);
    posP = posP > numeric_limits<Real>::max() ? numeric_limits<Real>::max() : posP;
    Real posW = k*t.data_prob / (posP + k*t.data_prob);

    Real log_pos_nc_p = (log_posP - Log<Real>::add(log_posP, log(k*t.data_prob)));
    log_pos_nc_p = log_pos_nc_p < -numeric_limits<Real>::max() ? -numeric_limits<Real>::max() : log_pos_nc_p;
    assert(abs(log_pos_nc_p) <= numeric_limits<Real>::max());
    f -= log_pos_nc_p;

    unnormalised_llh += log_posP;

    for (int i=0; i<context_width; i++) {
      weightedRepresentations.at(i).row(instance) -= posW * contributions(i) * model.R.row(w).transpose();
      g_R.row(w) -= posW * contributions(i) * prediction_vectors.at(i).row(instance).transpose();
      //g_M(i)     -= posW * (contributions(i) - pM(i));
      g_M(i)     -= (contributions(i) - pM(i));
    }
    g_B(w) -= posW;

    for (size_t nl_i=0; nl_i < t.noise_words.size(); ++nl_i) {
      WordId v_noise = t.noise_words.at(nl_i);

      for (int i=0; i<context_width; i++)
        logProbs(i) = model.R.row(v_noise) * prediction_vectors.at(i).row(instance).transpose() + model.B(v_noise);

      // calculate the mixture contributions
      mixLogProbs = logProbs.array() + log(pM.array());
      max = mixLogProbs.maxCoeff();
      contributions_log_z = log((mixLogProbs.array() - max).exp().sum()) + max;
      contributions = (mixLogProbs.array() - contributions_log_z).exp();

      Real log_negP = contributions_log_z;
      Real negP = exp(log_negP);

      negP = negP > numeric_limits<Real>::max() ? numeric_limits<Real>::max() : negP;
      Real negW = negP / (negP + k*t.noise_probs.at(nl_i));

      Real log_negNoise_nc_p = log(k*t.noise_probs.at(nl_i)) - Log<Real>::add(log_negP, log(k*t.noise_probs.at(nl_i)));
      log_negNoise_nc_p = log_negNoise_nc_p < -numeric_limits<Real>::max() ? -numeric_limits<Real>::max() : log_negNoise_nc_p;
      f -= log_negNoise_nc_p;

      for (int i=0; i<context_width; i++) {
        weightedRepresentations.at(i).row(instance) += negW * contributions(i) * model.R.row(v_noise).transpose();
        g_R.row(v_noise) += negW * contributions(i) * prediction_vectors.at(i).row(instance);
        //g_M(i)     += negW * (contributions(i) - pM(i));
      }
      g_B(v_noise) += negW;
    }

    /*
    // context gradient updates
    for (int i=0; i<context_width; ++i) {
    int j=t.data_index-context_width+i;
    int v_i = (j<0 ? start_id : training_corpus.at(j));
    g_Q.row(v_i) += model.context_product(i, weightedRepresentations.at(i).row(instance), true);

    model.context_gradient_update(g_C.at(i), context_vectors.at(i).row(instance), weightedRepresentations.at(i).row(instance));
    }
    */
  }

  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i=0; i<context_width; ++i) {
    context_gradients = model.context_product(i, weightedRepresentations.at(i), true); // weightedRepresentations*C(i)^T
    for (int instance=0; instance < instances; ++instance) {
      const TrainingInstance& t = training_instances.at(instance);
      int j=t.data_index-context_width+i;

      bool sentence_start = (j<0);
      for (int k=j; !sentence_start && k < t.data_index; k++)
        if (training_corpus.at(k) == end_id)
          sentence_start=true;
      int v_i = (sentence_start ? start_id : training_corpus.at(j));

      g_Q.row(v_i) += context_gradients.row(instance);
    }
    model.context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations.at(i));
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
    q_context_products.at(i) = model.context_product(i, model.Q);
    //q_context_products.at(i) = model.Q * model.C.at(i);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");
/*
  #pragma omp parallel \
      shared(test_corpus,model,stride,q_context_products,word_width) \
      reduction(+:p,log_z_sum,tokens)
*/
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

      ArrayReal score_vector = model.R * prediction_vector + model.B;
      Real w_p = score_vector(w);
      Real max_score = score_vector.maxCoeff();
      Real log_z = log((score_vector-max_score).exp().sum()) + max_score;
      w_p -= log_z;
      log_z_sum += log_z;
      p += w_p;

      #pragma omp master
      if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }

      tokens++;
    }
    #pragma omp master
    cerr << endl;
  }

  //return exp(-p/tokens);
  return p;
}

Real mixture_perplexity(const LogBiLinearModel& model, const Corpus& test_corpus, int stride) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;

  // cache the products of Q with the contexts
  std::vector<MatrixReal> q_context_products(context_width);
  for (int i=0; i<context_width; i++)
    q_context_products.at(i) = model.context_product(i, model.Q);

  // cache the mixture weights
  VectorReal pM = softMax(model.M);

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  #pragma omp master
  cerr << "Calculating perplexity for " << test_corpus.size()/stride << " tokens";

  for (size_t s = (thread_num*stride); s < test_corpus.size(); s += (num_threads*stride)) {
    WordId w = test_corpus.at(s);
    assert(w < model.R.rows() && "ERROR in mixture_perplexity(): word index out of range");

    int context_start = s - context_width;
    Real p_sum = 0;
    bool sentence_start = (s==0);
    //for (int i=0; i<context_width; ++i) {
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || test_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : test_corpus.at(j));
      assert(v_i < model.Q.rows() && "ERROR in mixture_perplexity(): context word index out of range");

      ArrayReal score_vector = model.R * q_context_products[i].row(v_i).transpose() + model.B;
      assert (softMax(score_vector).rows() > w);
      p_sum += (pM(i) * softMax(score_vector)(w));
    }

    p += log(p_sum);
    tokens++;

    #pragma omp master
    if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }
  }
  #pragma omp master
  cerr << endl;

  return p;
}
