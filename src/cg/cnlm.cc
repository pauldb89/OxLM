#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <math.h>
#include <iostream>
#include <functional>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#include "utils/conditional_omp.h"
#include "cg/cnlm.h"
#include "cg/utils.h"

using namespace std;
using namespace boost;
using namespace oxlm;


static boost::mt19937 linear_model_rng(static_cast<unsigned> (std::time(0)));
static uniform_01<> linear_model_uniform_dist;


CNLMBase::CNLMBase() : R(0,0,0), Q(0,0,0), F(0,0,0),
  B(0,0), FB(0,0), W(0,0), m_data(0) {}

CNLMBase::CNLMBase(const ModelData& config,
                               const Dict& target_labels,
                               const std::vector<int>& classes)
  : config(config), R(0,0,0), Q(0,0,0), F(0,0,0), B(0,0), FB(0,0),
  W(0,0), length_ratio(1), m_target_labels(target_labels), indexes(classes) {

  }

void CNLMBase::initWordToClass() {

  assert (!indexes.empty());
  word_to_class.clear();
  word_to_class.reserve(m_target_labels.size());
  for (int c = 0; c < int(indexes.size())-1; ++c) {
    int c_end = indexes.at(c+1);
    for (int w=indexes.at(c); w < c_end; ++w) {
      word_to_class.push_back(c);
    }
  }
  assert (m_target_labels.size() == word_to_class.size());
}

void CNLMBase::init(bool init_weights) {
  calculateDataSize(true);

  new (&W) WeightsType(m_data, m_data_size);
  if (init_weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i=0; i<m_data_size; i++)
      W(i) = gaussian(gen);
  }
  else W.setZero();

  Real* ptr = W.data();
  map_parameters(ptr, R, Q, F, C, B, FB);
}


int CNLMBase::calculateDataSize(bool allocate) {
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int F_size = config.classes * word_width;
  int C_size = context_width * (config.diagonal ? word_width : word_width*word_width);
  int B_size = num_output_words;
  int FB_size = config.classes;

  int data_size = R_size + Q_size + F_size + C_size + B_size + FB_size;
  if (allocate) {
    m_data_size = data_size;
    m_data = new Real[m_data_size];
  }
  return data_size;
}


void CNLMBase::hidden_layer(const std::vector<WordId>& context, const VectorReal& source, VectorReal& result) const {
  result = VectorReal::Zero(config.word_representation_size);
  int width = config.ngram_order-1;
  int gap = width-context.size();
  assert(static_cast<int>(context.size()) <= width);
  for (int i=gap; i < width; i++)
    if (m_target_labels.valid(context.at(i-gap))) {
      if (config.diagonal) result += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
      else                 result += Q.row(context.at(i-gap)) * C.at(i);
    }

  //////////////////////////////////////////////////////////////////
  // Source result contributions
  result += source;
  //////////////////////////////////////////////////////////////////

  // a simple non-linearity
  if (config.nonlinear)
    result = (1.0 + (-result).array().exp()).inverse(); // sigmoid
}

Real CNLMBase::log_prob(WordId w, const std::vector<WordId>& context, bool cache) const {
  const VectorReal s = VectorReal::Zero(config.word_representation_size);
  return log_prob(w, context, s, cache);
}

Real CNLMBase::log_prob(WordId w, const std::vector<WordId>& context, const VectorReal& source, bool cache) const {
  VectorReal prediction_vector;
  hidden_layer(context, source, prediction_vector);

  int c = get_class(w), c_start = indexes.at(c);

  VectorReal c_lps, w_lps;
  class_log_probs(context, source, prediction_vector, c_lps, cache); // p( . | context, source)
  word_log_probs(c, context, source, prediction_vector, w_lps, cache); // p( . | c, context, source)

  return c_lps(c) + w_lps(w - c_start);
}

void CNLMBase::class_log_probs(const std::vector<WordId>& context,
                                            const VectorReal& source, const VectorReal& prediction_vector,
                                            VectorReal& result,
                                            bool cache) const {
  // log p(c | context)
  // TODO(kmh): Fix caching to include source representation in cache.
  std::pair<std::unordered_map<Words, VectorReal, container_hash<Words> >::iterator, bool> context_cache_result;
  if (cache) context_cache_result = m_context_cache.insert(make_pair(context, VectorReal::Zero(config.classes)));
  if (cache && !context_cache_result.second) {
    result = context_cache_result.first->second;
  }
  else {
    Real c_log_z=0;
    result = logSoftMax(F*prediction_vector + FB, &c_log_z);
    assert(c_log_z != 0);
    if (cache) context_cache_result.first->second = result;
  }
}

void CNLMBase::word_log_probs(int c, const std::vector<WordId>& context,
                                           const VectorReal& source, const VectorReal& prediction_vector,
                                           VectorReal& result,
                                           bool cache) const {
  // log p(w | c, context)
  std::pair<std::unordered_map<std::pair<int,Words>, VectorReal>::iterator, bool> class_context_cache_result;
  if (cache) class_context_cache_result = m_context_class_cache.insert(make_pair(make_pair(c,context),VectorReal()));
  if (cache && !class_context_cache_result.second) {
    result = class_context_cache_result.first->second;
  }
  else {
    Real w_log_z=0;
    result = logSoftMax(class_R(c)*prediction_vector + class_B(c), &w_log_z);
    if (cache) class_context_cache_result.first->second = result;
  }
}

Real CNLMBase::gradient_(
    const std::vector<Sentence>& target_corpus,
    const TrainingInstances& training_instances,
    // std::function<void(TrainingInstance, VectorReal)> source_repr_callback,
    // std::function<void(TrainingInstance, int, int, VectorReal)> source_grad_callback,
    Real l2, Real source_l2, Real*& ptr) {
  WordVectorsType g_R(0,0,0), g_Q(0,0,0), g_F(0,0,0);
  ContextTransformsType g_C;
  WeightsType g_B(0,0), g_FB(0,0);
  map_parameters(ptr, g_R, g_Q, g_F, g_C, g_B, g_FB);
  // map_parameters_(g_W, &g_R, &g_Q, &g_F, &g_C, &g_B, &g_FB);

  Real f=0;
  WordId start_id = label_set().Convert("<s>");

  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int tokens=0;
  for (auto instance : training_instances)
    tokens += target_corpus.at(instance).size();

  //////////////////////////////////////////////////////////////////
  // LM prediction_vector contributions:
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
    const Sentence& target_sent = target_corpus.at(t);
    for (int t_i=0; t_i < int(target_sent.size()); ++t_i, ++instance_counter) {
      VectorReal s_vec = VectorReal::Zero(word_width);
      source_repr_callback(t, t_i, s_vec);
      prediction_vectors.row(instance_counter) += s_vec;
    }
  }
  //////////////////////////////////////////////////////////////////

  // the weighted sum of word representations
  MatrixReal weightedRepresentations = MatrixReal::Zero(tokens, word_width);

  // calculate the function and gradient for each ngram
  instance_counter=0;
  for (int instance=0; instance < instances; instance++) {
    const TrainingInstance& t = training_instances.at(instance);
    const Sentence& sent = target_corpus.at(t);
    for (int t_i=0; t_i < int(sent.size()); ++t_i, ++instance_counter) {
      WordId w = sent.at(t_i);

      int c = get_class(w);
      int c_start = indexes.at(c), c_end = indexes.at(c+1);

      if (!(w >= c_start && w < c_end))
        cerr << w << " " << c << " " << c_start << " " << c_end << endl;
      assert(w >= c_start && w < c_end);

      // a simple sigmoid non-linearity
      if (config.nonlinear) {
        prediction_vectors.row(instance_counter) = (1.0 + (-prediction_vectors.row(instance_counter)).array().exp()).inverse(); // sigmoid
        //for (int x=0; x<word_width; ++x)
        //  prediction_vector(x) *= (prediction_vector(x) > 0 ? 1 : 0.01); // rectifier
      }

      VectorReal class_conditional_scores = F * prediction_vectors.row(instance_counter).transpose() + FB;
      VectorReal word_conditional_scores  = class_R(c) * prediction_vectors.row(instance_counter).transpose() + class_B(c);

      ArrayReal class_conditional_log_probs = logSoftMax(class_conditional_scores);
      ArrayReal word_conditional_log_probs  = logSoftMax(word_conditional_scores);

      VectorReal class_conditional_probs = class_conditional_log_probs.exp();
      VectorReal word_conditional_probs  = word_conditional_log_probs.exp();

      // df/d(prediction_vectors)
      weightedRepresentations.row(instance_counter) -= (F.row(c) - class_conditional_probs.transpose() * F);
      weightedRepresentations.row(instance_counter) -= (R.row(w) - word_conditional_probs.transpose() * class_R(c));

      assert(isfinite(class_conditional_log_probs(c)));
      assert(isfinite(word_conditional_log_probs(w-c_start)));
      f -= (class_conditional_log_probs(c) + word_conditional_log_probs(w-c_start));

      // do the gradient updates:
      //   data contributions:
      g_F.row(c) -= prediction_vectors.row(instance_counter);
      g_R.row(w) -= prediction_vectors.row(instance_counter);
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

      //////////////////////////////////////////////////////////////////
      // Source word representations gradient
      source_grad_callback(t, t_i, instance_counter, weightedRepresentations.row(instance_counter));
      /*
       * if (window < 0) {
       *   for (auto s_i : source_sent)
       *     g_S.row(s_i) += weightedRepresentations.row(instance_counter);
       * }
       * else {
       *   int centre = min(floor(Real(t_i)*length_ratio + 0.5), double(source_len-1));
       *   int start = max(centre-window, 0);
       *   int end = min(source_len, centre+window+1);
       *   for (int i=start; i < end; ++i) {
       *     g_S.row(source_sent.at(i)) += window_product(i-centre+window, weightedRepresentations.row(instance_counter), true);
       *     context_gradient_update(g_T.at(i-centre+window), S.row(source_sent.at(i)), weightedRepresentations.row(instance_counter));
       *   }
       * }
       */
      //////////////////////////////////////////////////////////////////
    }
  }

  MatrixReal context_gradients = MatrixReal::Zero(word_width, tokens);
  for (int i=0; i<context_width; ++i) {
    context_gradients = context_product(i, weightedRepresentations, true); // weightedRepresentations*C(i)^T

    instance_counter=0;
    for (int instance=0; instance < instances; ++instance) {
      const TrainingInstance& t = training_instances.at(instance);
      const Sentence& sent = target_corpus.at(t);

      VectorReal sentence_weightedReps = VectorReal::Zero(word_width);
      for (int t_i=0; t_i < int(sent.size()); ++t_i, ++instance_counter) {
        int j = t_i-context_width+i;

        bool sentence_start = (j<0);
        int v_i = (sentence_start ? start_id : sent.at(j));

        g_Q.row(v_i) += context_gradients.row(instance_counter);
      }
    }
    context_gradient_update(g_C.at(i), context_vectors.at(i), weightedRepresentations);
  }

#pragma omp master
  {
    if (l2 > 0.0 || source_l2 > 0.0) {
      // l2 objective contributions
      f += (0.5*l2*(R.squaredNorm() + Q.squaredNorm() + B.squaredNorm() + F.squaredNorm() + FB.squaredNorm()));
      for (size_t c=0; c<C.size(); ++c)
        f += (0.5*l2*C.at(c).squaredNorm());

      // l2 gradient contributions
      if (config.updates.R)  g_R.array() += (l2*R.array());
      if (config.updates.Q)  g_Q.array() += (l2*Q.array());
      if (config.updates.F)  g_F.array() += (l2*F.array());
      if (config.updates.B)  g_B.array() += (l2*B.array());
      if (config.updates.FB)  g_FB.array() += (l2*FB.array());
      if (config.updates.C)
        for (size_t c=0; c<C.size(); ++c)
          g_C.at(c).array() += (l2*C.at(c).array());

    }
  }

  return f;
}

void CNLMBase::map_parameters(Real*& ptr, WordVectorsType& r, WordVectorsType& q, WordVectorsType& f,
                              ContextTransformsType& c, WeightsType& b, WeightsType& fb) const {
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int F_size = config.classes * word_width;
  int C_size = (config.diagonal ? word_width : word_width*word_width);

  new (&r) WordVectorsType(ptr, num_output_words, word_width);
  ptr += R_size;
  new (&q) WordVectorsType(ptr, num_context_words, word_width);
  ptr += Q_size;
  new (&f) WordVectorsType(ptr, config.classes, word_width);
  ptr += F_size;

  c.clear();
  for (int i=0; i<context_width; i++) {
    if (config.diagonal) c.push_back(ContextTransformType(ptr, word_width, 1));
    else                 c.push_back(ContextTransformType(ptr, word_width, word_width));
    ptr += C_size;
  }

  new (&b)  WeightsType(ptr, num_output_words);
  ptr += num_output_words;
  new (&fb) WeightsType(ptr, config.classes);
  ptr += config.classes;
}
