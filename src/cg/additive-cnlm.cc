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
#include "cg/additive-cnlm.h"
#include "cg/cnlm.h"
#include "cg/utils.h"

using namespace std;
using namespace boost;
using namespace oxlm;
using namespace std::placeholders;


void AdditiveCNLM::init(bool init_weights) {
  calculateDataSize(true);  // Calculates space requirements for this class and
                            //the parent and allocates space accordingly.

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
  map_parameters(ptr, R, Q, F, C, B, FB, S, T);
}

void AdditiveCNLM::source_representation(const Sentence& source,
                                         int target_index,
                                         VectorReal& result) const {
  result = VectorReal::Zero(config.word_representation_size);
  int window = config.source_window_width;

  if (target_index < 0 || window < 0) {
    for (auto s_i : source)
      result += S.row(s_i);
  }
  else {
    int source_len = source.size();
    int centre = min(floor(Real(target_index)*length_ratio + 0.5),
                     double(source_len-1));
    int start = max(centre-window, 0);
    int end = min(source_len, centre+window+1);

    for (int i=start; i < end; ++i)
      result += window_product(i-centre+window,
                               S.row(source.at(i))).transpose();
  }
}

Real AdditiveCNLM::log_prob(const WordId w, const std::vector<WordId>& context,
                            const Sentence& source, bool cache,
                            int target_index) const {
  VectorReal s;
  source_representation(source, target_index, s);
  return CNLMBase::log_prob(w, context, s, cache);
}

Real AdditiveCNLM::gradient(std::vector<Sentence>& source_corpus_,
                            const std::vector<Sentence>& target_corpus,
                            const TrainingInstances &training_instances,
                            Real l2, Real source_l2, WeightsType& g_W) {
#pragma omp master
  source_corpus = source_corpus_;

  Real* ptr = g_W.data();
  if (omp_get_thread_num() == 0)
    map_parameters(ptr, g_S, g_T);  // Allocates data for child.
  else { // TODO: come up with a better fix for this hack
    int word_width = config.word_representation_size;
    ptr += (source_types()*word_width)
           + g_T.size()*word_width*(config.diagonal ? 1 : word_width);
  }
#pragma omp barrier

  // Allocates data for parent.
  Real f = gradient_(target_corpus, training_instances, l2, source_l2, ptr);

  #pragma omp master
  {
    if (source_l2 > 0.0) {
      // l2 objective contributions
      f += (0.5*source_l2*S.squaredNorm());
      for (size_t t=0; t<T.size(); ++t)
        f += (0.5*source_l2*T.at(t).squaredNorm());

      // l2 gradient contributions
      if (config.updates.S)
        g_S.array() += (source_l2*S.array());
      if (config.updates.T)
        for (size_t t=0; t<T.size(); ++t)
          g_T.at(t).array() += (source_l2*T.at(t).array());
    }
  }
  return f;
}
