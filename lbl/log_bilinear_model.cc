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

LogBiLinearModel::LogBiLinearModel(const LogBiLinearModel& model) 
  : config(model.config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), m_labels(model.label_set()) {
    init(config, m_labels, false);
    addModel(model);
}

LogBiLinearModel::LogBiLinearModel(const ModelData& config, const Dict& labels)
  : config(config), R(0,0,0), Q(0,0,0), B(0,0), W(0,0), m_labels(labels) {
    init(config, m_labels, true);
}

void LogBiLinearModel::init(const ModelData& config, const Dict& labels, bool init_weights) {
    int num_words = labels.size();
    int word_width = config.word_representation_size;
    int context_width = config.ngram_order-1;

    int R_size = num_words*word_width;
    int Q_size = R_size;
    int C_size = word_width*word_width;
    int B_size = num_words;

    m_data_size = R_size + Q_size + context_width*C_size + B_size;
    m_data = new Real[m_data_size];

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

    new (&R) WordVectorsType(m_data, num_words, word_width);
    new (&Q) WordVectorsType(m_data+R_size, num_words, word_width);

    C.clear();
    Real* ptr = m_data+2*R_size;
    for (int i=0; i<context_width; i++) {
      C.push_back(ContextTransformType(ptr, word_width, word_width));
      ptr += C_size;
      //     C.back().setIdentity();
      //      C.back().setZero();
    }

    new (&B) WeightsType(ptr, num_words);

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

    assert(ptr+num_words== m_data+m_data_size); 

    #pragma omp master
    if (false) {
      std::cerr << "===============================" << std::endl;
      std::cerr << " Created a LogBiLinearModel: " << std::endl;
      std::cerr << "  Vocab size = " << num_words << std::endl;
      std::cerr << "  Word Vector size = " << word_width << std::endl;
      std::cerr << "  Context size = " << context_width << std::endl;
      std::cerr << "  Total parameters = " << m_data_size << std::endl;
      std::cerr << "===============================" << std::endl;
    }
  }
