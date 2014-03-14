#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#include "nlm.h"
#include "log_add.h"
#include "lbl/feature_generator.h"

using namespace std;
using namespace boost;
using namespace oxlm;

static std::mt19937 linear_model_rng(static_cast<unsigned> (time(0)));
static uniform_01<> linear_model_uniform_dist;

NLM::NLM()
    : R(0,0,0), Q(0,0,0), B(0,0), W(0,0), M(0,0),
      m_diagonal(false), m_data_size(0), m_data(NULL) {}

NLM::NLM(const NLM& model)
    : config(model.config), R(0, 0, 0), Q(0, 0, 0), B(0, 0), W(0, 0), M(0, 0),
      m_labels(model.m_labels), m_diagonal(model.m_diagonal),
      unigram(model.unigram), m_data_size(m_data_size) {
  cerr << "copiem cu spor" << endl;
  m_data = new Real[m_data_size];
  copy(model.m_data, model.m_data + m_data_size, m_data);
  initWeights(config, false);
}

NLM::NLM(const ModelData& config, const Dict& labels, bool diagonal)
    : config(config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), M(0,0),
      m_labels(labels), m_diagonal(diagonal) {
  init(config, config.random_weights);
}

void NLM::init(const ModelData& config, bool random_weights) {
  allocate_data(config);
  initWeights(config, random_weights);
}

void NLM::initWeights(const ModelData& config, bool random_weights) {
  // the prediction vector ranges over classes for a class based LM, or the vocab otherwise
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int C_size = (m_diagonal ? word_width : word_width*word_width);
  //int C_size;
  //if (m_diagonal) C_size = word_width;
  //else            C_size = word_width*word_width;

  new (&W) WeightsType(m_data, m_data_size);
  if (random_weights) {
    //    W.setRandom() /= 10;
    random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i = 0; i < m_data_size; i++) {
      W(i) = gaussian(gen);
    }
  } else {
    W.setZero();
  }

  new (&R) WordVectorsType(m_data, num_output_words, word_width);
  new (&Q) WordVectorsType(m_data+R_size, num_context_words, word_width);

  C.clear();
  Real* ptr = m_data+R_size+Q_size;
  for (int i=0; i<context_width; i++) {
    if (m_diagonal) C.push_back(ContextTransformType(ptr, word_width, 1));
    else            C.push_back(ContextTransformType(ptr, word_width, word_width));
    ptr += C_size;
    //     C.back().setIdentity();
    //      C.back().setZero();
  }

  new (&B) WeightsType(ptr, num_output_words);
  new (&M) WeightsType(ptr+num_output_words, context_width);

  M.setZero();

  //R.setOnes();
  //R.setZero();
  //R.setIdentity();
  //Q.setOnes();
  //Q.setIdentity();
  //Q.setZero();
  //B.setOnes();
  //    R << 0,0,0,1 , 0,0,1,0 , 0,1,0,0 , 1,0,0,0;
  //    R << 0,0 , 0,0 , 0,1 , 1,0;
  //    Q << 0,0,0,1 , 0,0,1,0 , 0,1,0,0 , 1,0,0,0;
  //    Q << 1,1 , 1,1 , 1,1 , 1,1;

  //  assert(ptr+num_output_words == m_data+m_data_size);

#pragma omp master
  if (true) {
    cerr << "===============================" << endl;
    cerr << " Created a NLM: "   << endl;
    cerr << "  Output Vocab size = "          << num_output_words << endl;
    cerr << "  Context Vocab size = "         << num_context_words << endl;
    cerr << "  Word Vector size = "           << word_width << endl;
    cerr << "  Context size = "               << context_width << endl;
    cerr << "  Diagonal = "                   << m_diagonal << endl;
    cerr << "  Total parameters = "           << m_data_size << endl;
    cerr << "===============================" << endl;
  }
}

void NLM::allocate_data(const ModelData& config) {
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int C_size = (m_diagonal ? word_width : word_width*word_width);
  int B_size = num_output_words;
  int M_size = context_width;

  m_data_size = R_size + Q_size + context_width*C_size + B_size + M_size;
  m_data = new Real[m_data_size];
}
