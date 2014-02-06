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


Real AdditiveCNLM::log_prob(const WordId w, const std::vector<WordId>& context,
                            const Sentence& source, bool cache,
                            int target_index) const {
  VectorReal s;
  source_representation(source, target_index, s);
  return CNLMBase::log_prob(w, context, s, cache);
}

