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
#include <boost/random.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "corpus/corpus.h"
#include "lbl/context_processor.h"
#include "lbl/factored_nlm.h"
#include "lbl/log_add.h"
#include "lbl/operators.h"
#include "lbl/word_to_class_index.h"

// Namespaces
using namespace boost;
using namespace std;
using namespace oxlm;
using namespace Eigen;

// TODO: Major refactoring needed, these methods belong to the model.

FactoredNLM learn(ModelData& config);

typedef int TrainingInstance;
typedef vector<TrainingInstance> TrainingInstances;
void scatter_data(int start, int end,
                  const vector<size_t>& indices,
                  TrainingInstances &result);

Real sgd_gradient(FactoredNLM& model,
                  const boost::shared_ptr<Corpus>& training_corpus,
                  const TrainingInstances &indexes,
                  const boost::shared_ptr<WordToClassIndex>& index,
                  WordVectorsType& g_R,
                  WordVectorsType& g_Q,
                  ContextTransformsType& g_C,
                  WeightsType& g_B,
                  MatrixReal & g_F,
                  VectorReal & g_FB);

Real perplexity(
    const FactoredNLM& model,
    const boost::shared_ptr<Corpus>& test_corpus);
void freq_bin_type(const std::string &corpus, int num_classes, std::vector<int>& classes, Dict& dict, VectorReal& class_bias);
void classes_from_file(const std::string &class_file, vector<int>& classes, Dict& dict, VectorReal& class_bias);

void displayStats(
    int minibatch_counter, Real& pp, Real& best_pp,
    const FactoredNLM& model,
    const boost::shared_ptr<Corpus>& test_corpus) {
  if (test_corpus != nullptr) {
    #pragma omp master
    pp = 0.0;

    // Each thread must wait until the perplexity is set to 0.
    // Otherwise, partial results might get overwritten.
    #pragma omp barrier

    Real local_pp = perplexity(model, test_corpus);
    #pragma omp critical
    pp += local_pp;

    // Wait for all threads to compute the perplexity for their slice of
    // test data.
    #pragma omp barrier
    #pragma omp master
    {
      pp = exp(-pp / test_corpus->size());
      cout << "\tMinibatch " << minibatch_counter
           << ", Test Perplexity = " << pp << endl;
      best_pp = min(best_pp, pp);
    }
  }
}

FactoredNLM learn(ModelData& config) {
  boost::shared_ptr<Corpus> training_corpus, test_corpus;
  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");

  //////////////////////////////////////////////
  // separate the word types into classes using
  // frequency binning
  vector<int> classes;
  VectorReal class_bias = VectorReal::Zero(config.classes);
  if (config.class_file.size()) {
    cerr << "--class-file set, ignoring --classes." << endl;
    classes_from_file(config.class_file, classes, dict, class_bias);
    config.classes = classes.size() - 1;
  } else {
    freq_bin_type(config.training_file, config.classes, classes,
                  dict, class_bias);
  }
  //////////////////////////////////////////////

 
  //////////////////////////////////////////////
  // read the training sentences
  ifstream in(config.training_file);
  string line, token;

  training_corpus = boost::make_shared<Corpus>();
  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token)
      training_corpus->push_back(dict.Convert(token));
    training_corpus->push_back(end_id);
  }
  in.close();
  //////////////////////////////////////////////

  //////////////////////////////////////////////
  // read the test sentences
  if (config.test_file.size()) {
    test_corpus = boost::make_shared<Corpus>();
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
        test_corpus->push_back(w);
      }
      test_corpus->push_back(end_id);
    }
    test_in.close();
  }
  //////////////////////////////////////////////

  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  FactoredNLM model(config, dict, index);
  model.FB = class_bias;

  if (config.model_input_file.size()) {
    std::ifstream f(config.model_input_file);
    boost::archive::text_iarchive ar(f);
    ar >> model;
  }

  vector<size_t> training_indices(training_corpus->size());
  model.unigram = VectorReal::Zero(model.labels());
  for (size_t i=0; i<training_indices.size(); i++) {
    model.unigram(training_corpus->at(i)) += 1;
    training_indices[i] = i;
  }
  model.B = ((model.unigram.array()+1.0)/(model.unigram.sum()+model.unigram.size())).log();
  model.unigram /= model.unigram.sum();

  VectorReal adaGrad = VectorReal::Zero(model.num_weights());
  VectorReal global_gradient(model.num_weights());
  Real av_f = 0;
  Real pp = 0, best_pp = numeric_limits<Real>::infinity();

  MatrixReal global_gradientF(model.F.rows(), model.F.cols());
  VectorReal global_gradientFB(model.FB.size());
  MatrixReal adaGradF = MatrixReal::Zero(model.F.rows(), model.F.cols());
  VectorReal adaGradFB = VectorReal::Zero(model.FB.size());

  omp_set_num_threads(config.threads);
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
    int C_size = config.diagonal_contexts ? word_width : word_width*word_width;
    int B_size = num_words;
    int M_size = context_width;

    assert((R_size+Q_size+context_width*C_size+B_size+M_size) == model.num_weights());

    Real* gradient_data = new Real[model.num_weights()];
    WeightsType gradient(gradient_data, model.num_weights());

    WordVectorsType g_R(gradient_data, num_words, word_width);
    WordVectorsType g_Q(gradient_data+R_size, num_words, word_width);

    ContextTransformsType g_C;
    Real* ptr = gradient_data+2*R_size;
    for (int i=0; i<context_width; i++) {
      if (config.diagonal_contexts) {
        g_C.push_back(ContextTransformType(ptr, word_width, 1));
      } else {
        g_C.push_back(ContextTransformType(ptr, word_width, word_width));
      }
      ptr += C_size;
    }

    WeightsType g_B(ptr, B_size);
    WeightsType g_M(ptr+B_size, M_size);
    MatrixReal g_F(num_classes, word_width);
    VectorReal g_FB(num_classes);
    //////////////////////////////////////////////

    size_t minibatch_counter = 1;
    size_t minibatch_size = config.minibatch_size;
    for (int iteration=0; iteration < config.iterations; ++iteration) {
      auto iteration_start = GetTime();
      #pragma omp master
      {
        av_f=0.0;
        pp=0.0;
        cout << "Iteration " << iteration << ": " << endl;

        if (config.randomise) {
          std::random_shuffle(training_indices.begin(), training_indices.end());
        }
      }

      TrainingInstances training_instances;
      Real step_size = config.step_size;

      for (size_t start=0; start < training_corpus->size() && (int)start < config.instances; ++minibatch_counter) {
        size_t end = min(training_corpus->size(), start + minibatch_size);

        #pragma omp master
        {
          global_gradient.setZero();
          global_gradientF.setZero();
          global_gradientFB.setZero();
        }

        gradient.setZero();
        g_F.setZero();
        g_FB.setZero();

        // Wait for global gradients to be cleared before updating them with
        // data from this minibatch.
        #pragma omp barrier
        scatter_data(start, end, training_indices, training_instances);
        Real f = sgd_gradient(model, training_corpus, training_instances, index,
                              g_R, g_Q, g_C, g_B, g_F, g_FB);

        #pragma omp critical
        {
          global_gradient += gradient;
          global_gradientF += g_F;
          global_gradientFB += g_FB;
          av_f += f;
        }

        // All global gradient updates must be completed before executing
        // adagrad updates.
        #pragma omp barrier
        #pragma omp master
        {
          adaGrad.array() += global_gradient.array().square();
          model.W -= global_gradient.binaryExpr(
              adaGrad, CwiseAdagradUpdateOp<Real>(step_size));

          adaGradF.array() += global_gradientF.array().square();
          model.F -= global_gradientF.binaryExpr(
              adaGradF, CwiseAdagradUpdateOp<Real>(step_size));
          adaGradFB.array() += global_gradientFB.array().square();
          model.FB -= global_gradientFB.binaryExpr(
              adaGradFB, CwiseAdagradUpdateOp<Real>(step_size));

          // regularisation
          if (config.l2_lbl > 0) {
            Real minibatch_factor = static_cast<Real>(end - start) / training_corpus->size();
            model.l2GradientUpdate(minibatch_factor);
            av_f += model.l2Objective(minibatch_factor);
          }
        }

        // Wait for master thread to update model.
        #pragma omp barrier

        if (minibatch_counter % 100 == 0) {
          displayStats(minibatch_counter, pp, best_pp, model, test_corpus);
        }

        start += minibatch_size;
      }

      displayStats(minibatch_counter, pp, best_pp, model, test_corpus);

      Real iteration_time = GetDuration(iteration_start, GetTime());
      #pragma omp master
      {
        cout << "Iteration: " << iteration << ", "
             << "Time: " << iteration_time << " seconds, "
             << "Average f = " << av_f / training_corpus->size() << endl;
        cout << endl;

        if (iteration >= 1 && config.reclass) {
          model.reclass(training_corpus, test_corpus);
          adaGradF = MatrixReal::Zero(model.F.rows(), model.F.cols());
          adaGradFB = VectorReal::Zero(model.FB.size());
          adaGrad = VectorReal::Zero(model.num_weights());
        }
      }

      if (config.model_output_file.size() && config.log_period) {
        if (iteration % config.log_period == 0) {
          string file = config.model_output_file + ".i" + to_string(iteration);
          cout << "Writing trained model to " << file << endl;
          std::ofstream f(file);
          boost::archive::text_oarchive ar(f);
          ar << model;
        }
      }
    }
  }

  cout << "Overall minimum perplexity: " << best_pp << endl;

  return model;
}


void scatter_data(int start, int end, const vector<size_t>& indices, TrainingInstances &result) {
  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  result.clear();
  result.reserve((end-start)/num_threads);

  for (int s = start+thread_num; s < end; s += num_threads) {
    result.push_back(indices.at(s));
  }
}


Real sgd_gradient(FactoredNLM& model,
                const boost::shared_ptr<Corpus>& training_corpus,
                const TrainingInstances &training_instances,
                const boost::shared_ptr<WordToClassIndex>& index,
                WordVectorsType& g_R,
                WordVectorsType& g_Q,
                ContextTransformsType& g_C,
                WeightsType& g_B,
                MatrixReal& g_F,
                VectorReal& g_FB) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");
  WordId end_id = model.label_set().Convert("</s>");
  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;
  ContextProcessor processor(training_corpus, context_width, start_id, end_id);

  // form matrices of the ngram histories
//  clock_t cache_start = clock();
  int instances=training_instances.size();
  vector<vector<WordId>> contexts(instances);
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width));
  for (int instance=0; instance < instances; ++instance) {
    contexts[instance] = processor.extract(training_instances[instance]);
    for (size_t i = 0; i < context_width; ++i) {
      context_vectors[i].row(instance) = model.Q.row(contexts[instance][i]);
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
  for (int instance=0; instance < instances; instance++) {
    int w_i = training_instances.at(instance);
    WordId w = training_corpus->at(w_i);
    int c = index->getClass(w);
    int c_start = index->getClassMarker(c);
    int c_size = index->getClassSize(c);
    int word_index = index->getWordIndexInClass(w);

    // a simple sigmoid non-linearity
    prediction_vectors.row(instance) = (1.0 + (-prediction_vectors.row(instance)).array().exp()).inverse(); // sigmoid
    //for (int x=0; x<word_width; ++x)
    //  prediction_vectors.row(instance)(x) *= (prediction_vectors.row(instance)(x) > 0 ? 1 : 0.01); // rectifier

    VectorReal class_conditional_scores = model.F * prediction_vectors.row(instance).transpose() + model.FB;
    VectorReal word_conditional_scores  = model.class_R(c) * prediction_vectors.row(instance).transpose() + model.class_B(c);

    ArrayReal class_conditional_log_probs = logSoftMax(class_conditional_scores);
    ArrayReal word_conditional_log_probs  = logSoftMax(word_conditional_scores);

    VectorReal class_conditional_probs = class_conditional_log_probs.exp();
    VectorReal word_conditional_probs  = word_conditional_log_probs.exp();

    weightedRepresentations.row(instance) -= (model.F.row(c) - class_conditional_probs.transpose() * model.F);
    weightedRepresentations.row(instance) -= (model.R.row(w) - word_conditional_probs.transpose() * model.class_R(c));

    assert(isfinite(class_conditional_log_probs(c)));
    assert(isfinite(word_conditional_log_probs(word_index)));
    f -= (class_conditional_log_probs(c) + word_conditional_log_probs(w-c_start));

    // do the gradient updates:
    //   data contributions:
    g_F.row(c) -= prediction_vectors.row(instance).transpose();
    g_R.row(w) -= prediction_vectors.row(instance).transpose();
    g_FB(c)    -= 1.0;
    g_B(w)     -= 1.0;
    //   model contributions:
    g_R.block(c_start, 0, c_size, g_R.cols()) += word_conditional_probs * prediction_vectors.row(instance);
    g_F += class_conditional_probs * prediction_vectors.row(instance);
    g_FB += class_conditional_probs;
    g_B.segment(c_start, c_size) += word_conditional_probs;

    // a simple sigmoid non-linearity
    weightedRepresentations.row(instance).array() *=
      prediction_vectors.row(instance).array() * (1.0 - prediction_vectors.row(instance).array()); // sigmoid
    //for (int x=0; x<word_width; ++x)
    //  weightedRepresentations.row(instance)(x) *= (prediction_vectors.row(instance)(x) > 0 ? 1 : 0.01); // rectifier
  }
//  clock_t iteration_time = clock() - iteration_start;

//  clock_t context_start = clock();
  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i = 0; i < context_width; ++i) {
    context_gradients = model.context_product(i, weightedRepresentations, true); // weightedRepresentations*C(i)^T
    for (int instance = 0; instance < instances; ++instance) {
      g_Q.row(contexts[instance][i]) += context_gradients.row(instance);
    }
    model.context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations);
  }
//  clock_t context_time = clock() - context_start;

  return f;
}

Real perplexity(const FactoredNLM& model, const boost::shared_ptr<Corpus>& test_corpus) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;
  int tokens = 0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");
  ContextProcessor processor(test_corpus, context_width, start_id, end_id);

  #pragma omp master
  cout << "Calculating perplexity for " << test_corpus->size() << " tokens";

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t s = thread_num; s < test_corpus->size(); s += num_threads) {
    vector<WordId> context = processor.extract(s);
    Real log_prob = model.log_prob(test_corpus->at(s), context, true, false);
    p += log_prob;

    #pragma omp master
    if (tokens % 10000 == 0) {
      cout << ".";
      cout.flush();
    }

    tokens++;
  }
  #pragma omp master
  cout << endl;

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
