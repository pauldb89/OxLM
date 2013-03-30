#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include "log_bilinear_model.h"
#include "log_add.h"

using namespace std;
using namespace boost;
using namespace oxlm;

static boost::mt19937 linear_model_rng(static_cast<unsigned> (std::time(0)));
static uniform_01<> linear_model_uniform_dist;

//LogBiLinearModel::LogBiLinearModel(const LogBiLinearModel& model) 
//  : config(model.config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), m_labels(model.label_set()) {
//    init(config, m_labels, false);
//    addModel(model);
//}

LogBiLinearModel::LogBiLinearModel(const ModelData& config, const Dict& labels)
  : config(config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), M(0,0), m_labels(labels) {
    init(config, m_labels, true);
}

void LogBiLinearModel::init(const ModelData& config, const Dict& labels, bool init_weights) {
  // the prediction vector ranges over classes for a class based LM, or the vocab otherwise
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int C_size = word_width*word_width;

  allocate_data(config);

  new (&W) WeightsType(m_data, m_data_size);
  if (init_weights) {
    //    W.setRandom() /= 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> gaussian(0,0.1);
    for (int i=0; i<m_data_size; i++)
      W(i) = gaussian(gen);
  }
  else W.setZero();

  new (&R) WordVectorsType(m_data, num_output_words, word_width);
  new (&Q) WordVectorsType(m_data+R_size, num_context_words, word_width);

  C.clear();
  Real* ptr = m_data+R_size+Q_size;
  for (int i=0; i<context_width; i++) {
    C.push_back(ContextTransformType(ptr, word_width, word_width));
    ptr += C_size;
    //     C.back().setIdentity();
    //      C.back().setZero();
  }

  new (&B) WeightsType(ptr, num_output_words);
  new (&M) WeightsType(ptr+num_output_words, context_width);

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
    std::cerr << "===============================" << std::endl;
    std::cerr << " Created a LogBiLinearModel: "   << std::endl;
    std::cerr << "  Output Vocab size = "          << num_output_words << std::endl;
    std::cerr << "  Context Vocab size = "         << num_context_words << std::endl;
    std::cerr << "  Word Vector size = "           << word_width << std::endl;
    std::cerr << "  Context size = "               << context_width << std::endl;
    std::cerr << "  Total parameters = "           << m_data_size << std::endl;
    std::cerr << "===============================" << std::endl;
  }
}

void LogBiLinearModel::allocate_data(const ModelData& config) {
  int num_output_words = output_types();
  int num_context_words = context_types();
  int word_width = config.word_representation_size;
  int context_width = config.ngram_order-1;

  int R_size = num_output_words * word_width;
  int Q_size = num_context_words * word_width;;
  int C_size = word_width*word_width;
  int B_size = num_output_words;
  int M_size = context_width;

  m_data_size = R_size + Q_size + context_width*C_size + B_size + M_size;
  m_data = new Real[m_data_size];
}

/*
LogBiLinearModelMixture::LogBiLinearModelMixture(const ModelData& c, const Dict& labels) : M(0,0) {
    config = c;
    m_labels = labels;
    init(config, m_labels, true);
    int context_width = config.ngram_order-1;
    new (&M) WeightsType(m_data+(m_data_size-context_width), context_width);
}

void LogBiLinearModelMixture::allocate_data(const ModelData& config) {
    int num_output_words = output_types();
    int num_context_words = context_types();
    int word_width = config.word_representation_size;
    int context_width = config.ngram_order-1;

    int R_size = num_output_words * word_width;
    int Q_size = num_context_words * word_width;;
    int C_size = word_width*word_width;
    int B_size = num_output_words;
    int M_size = context_width;

    m_data_size = R_size + Q_size + context_width*C_size + B_size + M_size;
    m_data = new Real[m_data_size];
}
*/

