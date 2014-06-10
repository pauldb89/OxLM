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
#include <set>

// Boost
#include <boost/random.hpp>
#include <boost/make_shared.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "corpus/corpus.h"
#include "lbl/context_processor.h"
#include "lbl/factored_nlm.h"
#include "lbl/log_add.h"
#include "lbl/model_utils.h"
#include "lbl/operators.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"
#include "utils/conditional_omp.h"

// Namespaces
using namespace boost;
using namespace std;
using namespace oxlm;
using namespace Eigen;

// TODO: Major refactoring needed, these methods belong to the model.

boost::shared_ptr<FactoredNLM> learn(ModelData& config);

typedef int TrainingInstance;
typedef vector<TrainingInstance> TrainingInstances;
Real sgd_gradient(const boost::shared_ptr<FactoredNLM>& model,
                  const boost::shared_ptr<Corpus>& training_corpus,
                  const TrainingInstances &indexes,
                  const boost::shared_ptr<WordToClassIndex>& index,
                  WordVectorsType& g_R,
                  WordVectorsType& g_Q,
                  ContextTransformsType& g_C,
                  WeightsType& g_B,
                  MatrixReal & g_F,
                  VectorReal & g_FB);

boost::shared_ptr<FactoredNLM> learn(ModelData& config) {
  boost::shared_ptr<Corpus> training_corpus, test_corpus;
  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");

  // Read classe from file or bin words according to freqeuncy.
  vector<int> classes;
  VectorReal class_bias;
  if (config.class_file.size()) {
    cout << "--class-file set, ignoring --classes." << endl;
    loadClassesFromFile(
        config.class_file, config.training_file, classes, dict, class_bias);
    config.classes = classes.size() - 1;
  } else {
    frequencyBinning(
        config.training_file, config.classes, classes, dict, class_bias);
  }

  training_corpus = readCorpus(config.training_file, dict);
  cout << "Done reading the training data..." << endl;
  if (config.test_file.size()) {
    test_corpus = readCorpus(config.test_file, dict);
    cout << "Done reading the test data..." << endl;
  }

  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);

  boost::shared_ptr<FactoredNLM> model;
  if (config.model_input_file.size()) {
    model = loadModel(config.model_input_file, test_corpus);
  } else {
    model = boost::make_shared<FactoredNLM>(config, dict, index);
    model->FB = class_bias;
  }

  vector<int> training_indices(training_corpus->size());
  model->unigram = VectorReal::Zero(model->labels());
  for (size_t i=0; i<training_indices.size(); i++) {
    model->unigram(training_corpus->at(i)) += 1;
    training_indices[i] = i;
  }
  model->B = ((model->unigram.array()+1.0)/(model->unigram.sum()+model->unigram.size())).log();
  model->unigram /= model->unigram.sum();

  VectorReal adaGrad = VectorReal::Zero(model->num_weights());
  VectorReal global_gradient(model->num_weights());
  Real av_f = 0;
  Real pp = 0, best_pp = numeric_limits<Real>::infinity();

  MatrixReal global_gradientF(model->F.rows(), model->F.cols());
  VectorReal global_gradientFB(model->FB.size());
  MatrixReal adaGradF = MatrixReal::Zero(model->F.rows(), model->F.cols());
  VectorReal adaGradFB = VectorReal::Zero(model->FB.size());

  omp_set_num_threads(config.threads);
  #pragma omp parallel shared(global_gradient, global_gradientF)
  {
    //////////////////////////////////////////////
    // setup the gradient matrices
    int num_words = model->labels();
    int num_classes = model->config.classes;
    int word_width = model->config.word_representation_size;
    int context_width = model->config.ngram_order-1;

    int R_size = num_words*word_width;
    int Q_size = R_size;
    int C_size = config.diagonal_contexts ? word_width : word_width*word_width;
    int B_size = num_words;
    int M_size = context_width;

    assert((R_size+Q_size+context_width*C_size+B_size+M_size) == model->num_weights());

    Real* gradient_data = new Real[model->num_weights()];
    WeightsType gradient(gradient_data, model->num_weights());

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
        vector<int> training_instances =
            scatterMinibatch(start, end, training_indices);
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
          model->W -= global_gradient.binaryExpr(
              adaGrad, CwiseAdagradUpdateOp<Real>(step_size));

          adaGradF.array() += global_gradientF.array().square();
          model->F -= global_gradientF.binaryExpr(
              adaGradF, CwiseAdagradUpdateOp<Real>(step_size));
          adaGradFB.array() += global_gradientFB.array().square();
          model->FB -= global_gradientFB.binaryExpr(
              adaGradFB, CwiseAdagradUpdateOp<Real>(step_size));

          // regularisation
          if (config.l2_lbl > 0) {
            Real minibatch_factor = static_cast<Real>(end - start) / training_corpus->size();
            model->l2GradientUpdate(minibatch_factor);
            av_f += model->l2Objective(minibatch_factor);
          }
        }

        // Wait for master thread to update model.
        #pragma omp barrier

        if ((minibatch_counter % 100 == 0 && minibatch_counter <= 1000) ||
            minibatch_counter % 1000 == 0) {
          evaluateModel(config, model, test_corpus, minibatch_counter,
                        iteration_start, pp, best_pp);
        }

        start += minibatch_size;
      }

      evaluateModel(config, model, test_corpus, minibatch_counter,
                    iteration_start, pp, best_pp);

      Real iteration_time = GetDuration(iteration_start, GetTime());
      #pragma omp master
      {
        cout << "Iteration: " << iteration << ", "
             << "Time: " << iteration_time << " seconds, "
             << "Average f = " << av_f / training_corpus->size() << endl;
        cout << endl;

        if (iteration >= 1 && config.reclass) {
          model->reclass(training_corpus, test_corpus);
          adaGradF = MatrixReal::Zero(model->F.rows(), model->F.cols());
          adaGradFB = VectorReal::Zero(model->FB.size());
          adaGrad = VectorReal::Zero(model->num_weights());
        }
      }
    }
  }

  cout << "Overall minimum perplexity: " << best_pp << endl;

  return model;
}

Real sgd_gradient(
    const boost::shared_ptr<FactoredNLM>& model,
    const boost::shared_ptr<Corpus>& training_corpus,
    const TrainingInstances &training_instances,
    const boost::shared_ptr<WordToClassIndex>& index,
    WordVectorsType& g_R, WordVectorsType& g_Q,
    ContextTransformsType& g_C, WeightsType& g_B,
    MatrixReal& g_F, VectorReal& g_FB) {
  Real f=0;
  WordId start_id = model->label_set().Convert("<s>");
  WordId end_id = model->label_set().Convert("</s>");
  int word_width = model->config.word_representation_size;
  int context_width = model->config.ngram_order-1;
  ContextProcessor processor(training_corpus, context_width, start_id, end_id);

  // form matrices of the ngram histories
//  clock_t cache_start = clock();
  int instances=training_instances.size();
  vector<vector<WordId>> contexts(instances);
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width));
  for (int instance=0; instance < instances; ++instance) {
    contexts[instance] = processor.extract(training_instances[instance]);
    for (size_t i = 0; i < context_width; ++i) {
      context_vectors[i].row(instance) = model->Q.row(contexts[instance][i]);
    }
  }
  MatrixReal prediction_vectors = MatrixReal::Zero(instances, word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += model->context_product(i, context_vectors.at(i));

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

    VectorReal class_conditional_scores = model->F * prediction_vectors.row(instance).transpose() + model->FB;
    VectorReal word_conditional_scores  = model->class_R(c) * prediction_vectors.row(instance).transpose() + model->class_B(c);

    ArrayReal class_conditional_log_probs = logSoftMax(class_conditional_scores);
    ArrayReal word_conditional_log_probs  = logSoftMax(word_conditional_scores);

    VectorReal class_conditional_probs = class_conditional_log_probs.exp();
    VectorReal word_conditional_probs  = word_conditional_log_probs.exp();

    weightedRepresentations.row(instance) -= (model->F.row(c) - class_conditional_probs.transpose() * model->F);
    weightedRepresentations.row(instance) -= (model->R.row(w) - word_conditional_probs.transpose() * model->class_R(c));

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
    context_gradients = model->context_product(i, weightedRepresentations, true); // weightedRepresentations*C(i)^T
    for (int instance = 0; instance < instances; ++instance) {
      g_Q.row(contexts[instance][i]) += context_gradients.row(instance);
    }
    model->context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations);
  }
//  clock_t context_time = clock() - context_start;

  return f;
}
