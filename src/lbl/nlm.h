#pragma once

#include <iostream>
#include <fstream>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"
#include "lbl/nlm_approximate_z.h"

namespace oxlm {

typedef Eigen::Map<MatrixReal> ContextTransformType;
typedef vector<ContextTransformType> ContextTransformsType;
typedef Eigen::Map<MatrixReal> WordVectorsType;
typedef Eigen::Map<VectorReal> WeightsType;

class NLM {
 public:
  NLM(const ModelData& config, const Dict& labels, bool diagonal=false);

  NLM(const NLM& model);

  virtual ~NLM() { delete [] m_data; }

  int output_types() const { return m_labels.size(); }
  int context_types() const { return m_labels.size(); }

  int labels() const { return m_labels.size(); }
  const Dict& label_set() const { return m_labels; }
  Dict& label_set() { return m_labels; }

  virtual void l2GradientUpdate(Real minibatch_factor);

  virtual Real l2Objective(Real minibatch_factor) const;

  void addModel(const NLM& model) { W += model.W; };

  void divide(Real d) { W /= d; }

  WordId label_id(const Word& l) const { return m_labels.Lookup(l); }

  const Word& label_str(WordId i) const { return m_labels.Convert(i); }

  int num_weights() const { return m_data_size; }

  Real* data() { return m_data; }

  virtual Real
  score(const WordId w, const vector<WordId>& context, const NLMApproximateZ& z_approx) const {
    VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
    int width = config.ngram_order-1;
    int gap = width-context.size();
    assert(static_cast<int>(context.size()) <= width);
    for (int i=gap; i < width; i++)
      if (m_diagonal) prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
      else            prediction_vector += Q.row(context.at(i-gap)) * C.at(i);
      //prediction_vector += context_product(i, Q.row(context.at(i-gap)).transpose());
    //return R.row(w) * prediction_vector + B(w);// - z_approx.z(prediction_vector);
    Real psi = R.row(w) * prediction_vector + B(w);
//    Real log_uw = log(unigram);
    return psi - log(exp(psi) + unigram(w));
  }

  virtual Real
  log_prob(const WordId w, const vector<WordId>& context) const {
    VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
    int width = config.ngram_order-1;
    int gap = width-context.size();
    assert(static_cast<int>(context.size()) <= width);
    for (int i=gap; i < width; i++)
      if (m_diagonal) prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
      else            prediction_vector += Q.row(context.at(i-gap)) * C.at(i);

    VectorReal word_probs = logSoftMax((R*prediction_vector).array() + B(w));
    return word_probs(w);
  }

  MatrixReal context_product(int i, const MatrixReal& v, bool transpose=false) const {
    if (m_diagonal)     {
      return (C.at(i).asDiagonal() * v.transpose()).transpose();
    }
    else if (transpose) return v * C.at(i).transpose();
    else                return v * C.at(i);
  }

  void context_gradient_update(ContextTransformType& g_C, const MatrixReal& v,const MatrixReal& w) const {
    if (m_diagonal) g_C += (v.cwiseProduct(w).colwise().sum()).transpose();
    else            g_C += (v.transpose() * w);
  }

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar << config;
    ar << m_labels;
    ar << m_diagonal;
    ar << boost::serialization::make_array(m_data, m_data_size);

    int unigram_len=unigram.rows();
    ar << unigram_len;
    ar << boost::serialization::make_array(unigram.data(), unigram_len);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> config;
    ar >> m_labels;
    ar >> m_diagonal;
    delete [] m_data;
    init(config, false);
    ar >> boost::serialization::make_array(m_data, m_data_size);

    int unigram_len=0;
    ar >> unigram_len;
    unigram = VectorReal(unigram_len);
    ar >> boost::serialization::make_array(unigram.data(), unigram_len);
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

 public:
  ModelData config;

  ContextTransformsType C;
  WordVectorsType       R;
  WordVectorsType       Q;
  WeightsType           B;
  WeightsType           W;
  VectorReal            unigram;

 protected:
  NLM();

  virtual void init(const ModelData& config, bool random_weights);
  virtual void initWeights(const ModelData& config, bool random_weights);
  virtual void allocate_data(const ModelData& config);

  Dict m_labels;
  int m_data_size;
  Real* m_data;
  bool m_diagonal;
};

} // namespace oxlm
