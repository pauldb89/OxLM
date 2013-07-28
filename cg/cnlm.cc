#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#include "cnlm.h"
#include "utils.h"


using namespace std;
using namespace boost;
using namespace oxlm;


static boost::mt19937 linear_model_rng(static_cast<unsigned> (std::time(0)));
static uniform_01<> linear_model_uniform_dist;


ConditionalNLM::ConditionalNLM(const ModelData& config, 
                                     const Dict& source_labels, 
                                     const Dict& target_labels, 
                                     const std::vector<int>& classes) 
  : config(config), R(0,0,0), Q(0,0,0), F(0,0,0), S(0,0,0), B(0,0), FB(0,0), W(0,0), 
    m_source_labels(source_labels), m_target_labels(target_labels), indexes(classes) {
    init(true);

    assert (!classes.empty());
    word_to_class.reserve(m_target_labels.size());
    for (int c=0; c < int(classes.size())-1; ++c) {
      int c_end = classes.at(c+1);
      //cerr << "\nClass " << c << ":" << endl;
      for (int w=classes.at(c); w < c_end; ++w) {
        word_to_class.push_back(c);
        //cerr << " " << label_str(w);
      }
    }
    assert (m_target_labels.size() == word_to_class.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i=0; i<F.rows(); i++) {
      FB(i) = gaussian(gen);
      for (int j=0; j<F.cols(); j++)
        F(i,j) = gaussian(gen);
    }
  }

void ConditionalNLM::init(bool init_weights) {
  allocate_data();

  new (&W) WeightsType(m_data, m_data_size);
  if (init_weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i=0; i<m_data_size; i++)
      W(i) = gaussian(gen);
  }
  else W.setZero();

  map_parameters(W, R, Q, F, S, C, B, FB); 
/*
#pragma omp master
  if (true) {
    std::cerr << "===============================" << std::endl;
    std::cerr << " Created a NLM: "   << std::endl;
    std::cerr << "  Output Vocab size = "          << num_output_words << std::endl;
    std::cerr << "  Context Vocab size = "         << num_context_words << std::endl;
    std::cerr << "  Word Vector size = "           << word_width << std::endl;
    std::cerr << "  Context size = "               << context_width << std::endl;
    std::cerr << "  Diagonal = "                   << m_diagonal << std::endl;
    std::cerr << "  Total parameters = "           << m_data_size << std::endl;
    std::cerr << "===============================" << std::endl;
  }
*/
}


void ConditionalNLM::allocate_data() {
  int num_source_words = source_types();
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int F_size = config.classes * word_width;;
  int S_size = num_source_words * word_width;;
  int C_size = (config.diagonal ? word_width : word_width*word_width);
  int B_size = num_output_words;
  int FB_size = config.classes;

  m_data_size = R_size + Q_size + F_size + S_size + context_width*C_size + B_size + FB_size;
  m_data = new Real[m_data_size];
}


Real ConditionalNLM::log_prob(const WordId w, const std::vector<WordId>& context, const Sentence& source, bool cache) const {
  VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
  int width = config.ngram_order-1;
  int gap = width-context.size();
  assert(static_cast<int>(context.size()) <= width);
  for (int i=gap; i < width; i++)
    if (config.diagonal) prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
    else                 prediction_vector += Q.row(context.at(i-gap)) * C.at(i);

  int c = get_class(w);

  //////////////////////////////////////////////////////////////////
  // Source prediction_vector contributions
  for (auto s_i : source)
    prediction_vector += S.row(s_i);
  //////////////////////////////////////////////////////////////////

  // a simple non-linearity
  if (config.nonlinear)
    prediction_vector = (1.0 + (-prediction_vector).array().exp()).inverse(); // sigmoid

  // log p(c | context) 
  Real class_log_prob = 0;
  std::pair<std::unordered_map<Words, Real, container_hash<Words> >::iterator, bool> context_cache_result;
  if (cache) context_cache_result = m_context_cache.insert(make_pair(context,0));
  if (cache && !context_cache_result.second) {
    assert (context_cache_result.first->second != 0);
    class_log_prob = F.row(c)*prediction_vector + FB(c) - context_cache_result.first->second;
  }
  else {
    Real c_log_z=0;
    VectorReal class_probs = logSoftMax(F*prediction_vector + FB, &c_log_z);
    assert(c_log_z != 0);
    class_log_prob = class_probs(c);
    if (cache) context_cache_result.first->second = c_log_z;
  }

  // log p(w | c, context) 
  Real word_log_prob = 0;
  std::pair<std::unordered_map<std::pair<int,Words>, Real>::iterator, bool> class_context_cache_result;
  if (cache) class_context_cache_result = m_context_class_cache.insert(make_pair(make_pair(c,context),0));
  if (cache && !class_context_cache_result.second) {
    word_log_prob  = R.row(w)*prediction_vector + B(w) - class_context_cache_result.first->second;
  }
  else {
    int c_start = indexes.at(c);
    Real w_log_z=0;
    VectorReal word_probs = logSoftMax(class_R(c)*prediction_vector + class_B(c), &w_log_z);
    word_log_prob = word_probs(w-c_start);
    if (cache) class_context_cache_result.first->second = w_log_z;
  }

  return class_log_prob + word_log_prob;
}


Real ConditionalNLM::gradient(const std::vector<Sentence>& source_corpus, const std::vector<Sentence>& target_corpus, 
                              const TrainingInstances &training_instances,
                              Real lambda, WeightsType& g_W) {
  WordVectorsType g_R(0,0,0), g_Q(0,0,0), g_F(0,0,0), g_S(0,0,0);
  ContextTransformsType g_C;
  WeightsType g_B(0,0), g_FB(0,0);
  map_parameters(g_W, g_R, g_Q, g_F, g_S, g_C, g_B, g_FB);

  Real f=0;
  WordId start_id = label_set().Convert("<s>");

  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int tokens=0;
  for (auto instance : training_instances)
    tokens += target_corpus.at(instance).size();

  //////////////////////////////////////////////////////////////////
  // LM prediction_vector contributions
  // form matrices of the ngram histories
  //  clock_t cache_start = clock();
  int instances=training_instances.size();
  int instance_counter=0;
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(tokens, word_width)); 
  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    const Sentence& sent = target_corpus.at(t);
    for (int s_i=0; s_i < int(sent.size()); ++s_i, ++instance_counter) {
      int context_start = s_i - context_width;

      bool sentence_start = (s_i==0);
      for (int i=context_width-1; i>=0; --i) {
        int j=context_start+i;
        sentence_start = (sentence_start || j<0);
        int v_i = (sentence_start ? start_id : sent.at(j));
        context_vectors.at(i).row(instance_counter) = Q.row(v_i);
      }
    }
  }
  MatrixReal prediction_vectors = MatrixReal::Zero(tokens, word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += context_product(i, context_vectors.at(i));
  //////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////
  // Source prediction_vector contributions
  instance_counter=0;
  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    VectorReal s_vec = VectorReal::Zero(word_width);
    for (auto s_i : source_corpus.at(t))
      s_vec += S.row(s_i);

    const Sentence& target_sent = target_corpus.at(t);
    for (int t_i=0; t_i < int(target_sent.size()); ++t_i, ++instance_counter) {
      prediction_vectors.row(instance_counter) += s_vec;
    }
  }
  //////////////////////////////////////////////////////////////////

  //  clock_t cache_time = clock() - cache_start;

  // the weighted sum of word representations
  MatrixReal weightedRepresentations = MatrixReal::Zero(tokens, word_width);

  // calculate the function and gradient for each ngram
  //  clock_t iteration_start = clock();
  instance_counter=0;
  for (int instance=0; instance < instances; instance++) {
    const TrainingInstance& t = training_instances.at(instance);
    const Sentence& sent = target_corpus.at(t);
    for (int s_i=0; s_i < int(sent.size()); ++s_i, ++instance_counter) {
      WordId w = sent.at(s_i);

      int c = get_class(w);
      int c_start = indexes.at(c), c_end = indexes.at(c+1);

      if (!(w >= c_start && w < c_end))
        cerr << w << " " << c << " " << c_start << " " << c_end << endl;
      assert(w >= c_start && w < c_end);

      // a simple sigmoid non-linearity
      if (config.nonlinear) {
        prediction_vectors.row(instance_counter) = (1.0 + (-prediction_vectors.row(instance_counter)).array().exp()).inverse(); // sigmoid
        //for (int x=0; x<word_width; ++x)
        //  prediction_vectors.row(instance_counter)(x) *= (prediction_vectors.row(instance_counter)(x) > 0 ? 1 : 0.01); // rectifier
      }

      VectorReal class_conditional_scores = F * prediction_vectors.row(instance_counter).transpose() + FB;
      VectorReal word_conditional_scores  = class_R(c) * prediction_vectors.row(instance_counter).transpose() + class_B(c);

      ArrayReal class_conditional_log_probs = logSoftMax(class_conditional_scores);
      ArrayReal word_conditional_log_probs  = logSoftMax(word_conditional_scores);

      VectorReal class_conditional_probs = class_conditional_log_probs.exp();
      VectorReal word_conditional_probs  = word_conditional_log_probs.exp();

      weightedRepresentations.row(instance_counter) -= (F.row(c) - class_conditional_probs.transpose() * F);
      weightedRepresentations.row(instance_counter) -= (R.row(w) - word_conditional_probs.transpose() * class_R(c));

      assert(isfinite(class_conditional_log_probs(c)));
      assert(isfinite(word_conditional_log_probs(w-c_start)));
      f -= (class_conditional_log_probs(c) + word_conditional_log_probs(w-c_start));

      // do the gradient updates:
      //   data contributions: 
      g_F.row(c) -= prediction_vectors.row(instance_counter).transpose();
      g_R.row(w) -= prediction_vectors.row(instance_counter).transpose();
      g_FB(c)    -= 1.0;
      g_B(w)     -= 1.0;
      //   model contributions: 
      g_R.block(c_start, 0, c_end-c_start, g_R.cols()) += word_conditional_probs * prediction_vectors.row(instance_counter);
      g_F += class_conditional_probs * prediction_vectors.row(instance_counter);
      g_FB += class_conditional_probs;
      g_B.segment(c_start, c_end-c_start) += word_conditional_probs;

      // a simple sigmoid non-linearity
      if (config.nonlinear) {
        weightedRepresentations.row(instance_counter).array() *= 
          prediction_vectors.row(instance_counter).array() * (1.0 - prediction_vectors.row(instance_counter).array()); // sigmoid
        //for (int x=0; x<word_width; ++x)
        //  weightedRepresentations.row(instance_counter)(x) *= (prediction_vectors.row(instance_counter)(x) > 0 ? 1 : 0.01); // rectifier
      }
    }
  }
  //  clock_t iteration_time = clock() - iteration_start;

  //  clock_t context_start = clock();
  MatrixReal context_gradients = MatrixReal::Zero(word_width, tokens);
  for (int i=0; i<context_width; ++i) {
    context_gradients = context_product(i, weightedRepresentations, true); // weightedRepresentations*C(i)^T

    instance_counter=0;
    for (int instance=0; instance < instances; ++instance) {
      const TrainingInstance& t = training_instances.at(instance);
      const Sentence& sent = target_corpus.at(t);
      VectorReal sentence_weightedReps = VectorReal::Zero(word_width);
      for (int s_i=0; s_i < int(sent.size()); ++s_i, ++instance_counter) {
        int j = s_i-context_width+i;

        bool sentence_start = (j<0);
        int v_i = (sentence_start ? start_id : sent.at(j));

        g_Q.row(v_i) += context_gradients.row(instance_counter);

        sentence_weightedReps += weightedRepresentations.row(instance_counter);
      }

      //////////////////////////////////////////////////////////////////
      // Source word representations gradient
      for (auto s_i : source_corpus.at(t))
        g_S.row(s_i) += sentence_weightedReps;
      //////////////////////////////////////////////////////////////////
    }
    context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations);
  }
  //  clock_t context_time = clock() - context_start;

  return f;
}

void ConditionalNLM::map_parameters(WeightsType& w, WordVectorsType& r, WordVectorsType& q, WordVectorsType& f, 
                                    WordVectorsType& s, ContextTransformsType& c, WeightsType& b, WeightsType& fb) const {
  int num_source_words = source_types();
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int F_size = config.classes * word_width;
  int S_size = num_source_words * word_width;
  int C_size = (config.diagonal ? word_width : word_width*word_width);

  Real* ptr = w.data();

  new (&r) WordVectorsType(ptr, num_output_words, word_width);
  ptr += R_size;
  new (&q) WordVectorsType(ptr, num_context_words, word_width);
  ptr += Q_size;
  new (&f) WordVectorsType(ptr, config.classes, word_width);
  ptr += F_size;
  new (&s) WordVectorsType(ptr, num_source_words, word_width);
  ptr += S_size;

  c.clear();
  for (int i=0; i<context_width; i++) {
    if (config.diagonal) c.push_back(ContextTransformType(ptr, word_width, 1));
    else                 c.push_back(ContextTransformType(ptr, word_width, word_width));
    ptr += C_size;
  }

  new (&b)  WeightsType(ptr, num_output_words);
  ptr += num_output_words;
  new (&fb) WeightsType(ptr, config.classes);
}
