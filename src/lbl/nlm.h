#ifndef _NLM_H_
#define _NLM_H_

#include <iostream>
#include <fstream>
#include <memory>

#include <Eigen/Dense>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/feature_store.h"
#include "lbl/utils.h"

namespace oxlm {

class NLMApproximateZ {
public:
  NLMApproximateZ() {}

  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    int m_z_approx_rows=m_z_approx.rows(), m_z_approx_cols=m_z_approx.cols();
    ar << m_z_approx_rows; 
    ar << m_z_approx_cols;
	  ar << boost::serialization::make_array(m_z_approx.data(), m_z_approx.rows() * m_z_approx.cols());
	  ar << boost::serialization::make_array(m_b_approx.data(), m_b_approx.rows());
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    int m_z_approx_rows=0, m_z_approx_cols=0;
    ar >> m_z_approx_rows; ar >> m_z_approx_cols;

    m_z_approx = MatrixReal(m_z_approx_rows, m_z_approx_cols);
    m_b_approx = VectorReal(m_z_approx_cols);

	  ar >> boost::serialization::make_array(m_z_approx.data(), m_z_approx.rows() * m_z_approx.cols());
	  ar >> boost::serialization::make_array(m_b_approx.data(), m_b_approx.rows());
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  Real z(const VectorReal& context) const {
    VectorReal z_products = context.transpose()*m_z_approx + m_b_approx.transpose(); // 1 x Z
    Real row_max = z_products.maxCoeff(); // 1 x 1
    VectorReal exp_z_products = (z_products.array() - row_max).exp(); // 1 x Z
    return log(exp_z_products.sum()) + row_max; // 1 x 1
  }

  void train(const MatrixReal& contexts, const VectorReal& zs, 
             Real step_size, int iterations, int approx_vectors);
  void train_lbfgs(const MatrixReal& contexts, const VectorReal& zs, 
                   Real step_size, int iterations, int approx_vectors);

private:
  MatrixReal m_z_approx;
  VectorReal m_b_approx;
};


class NLM {
public:
  typedef Eigen::Map<MatrixReal> ContextTransformType;
  typedef std::vector<ContextTransformType> ContextTransformsType;
  typedef Eigen::Map<MatrixReal> WordVectorsType;
  typedef Eigen::Map<VectorReal> WeightsType;

public:
  NLM(const ModelData& config, const Dict& labels, bool diagonal=false);
//  NLM(const NLM& model);

  virtual ~NLM() { delete [] m_data; }

  //int output_types() const { return config.classes > 0 ? config.classes : m_labels.size(); }
  int output_types() const { return m_labels.size(); }
  int context_types() const { return m_labels.size(); }

  int labels() const { return m_labels.size(); }
  const Dict& label_set() const { return m_labels; }
  Dict& label_set() { return m_labels; }

  virtual Real l2_gradient_update(Real sigma) { 
    W -= W*sigma; 
    return W.array().square().sum();
  }

  void addModel(const NLM& model) { W += model.W; };

  void divide(Real d) { W /= d; }

  WordId label_id(const Word& l) const { return m_labels.Lookup(l); }

  const Word& label_str(WordId i) const { return m_labels.Convert(i); }

  int num_weights() const { return m_data_size; }

  Real* data() { return m_data; }

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
    init(config, m_labels, false);
    ar >> boost::serialization::make_array(m_data, m_data_size);

    int unigram_len=0;
    ar >> unigram_len;
    unigram = VectorReal(unigram_len);
    ar >> boost::serialization::make_array(unigram.data(), unigram_len);
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  virtual Real 
  score(const WordId w, const std::vector<WordId>& context, const NLMApproximateZ& z_approx) const {
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
  log_prob(const WordId w, const std::vector<WordId>& context) const {
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

public:
  ModelData config;

  ContextTransformsType C;
  WordVectorsType       R;
  WordVectorsType       Q;
  WeightsType           B;
  WeightsType           W;
  WeightsType           M;
  VectorReal            unigram;

protected:
  NLM();

  virtual void init(const ModelData& config, const Dict& labels, bool init_weights=false);
  virtual void allocate_data(const ModelData& config);

  Dict m_labels;
  int m_data_size;
  Real* m_data;
  bool m_diagonal;
};


class FactoredOutputNLM: public NLM {
public:
  FactoredOutputNLM();

  FactoredOutputNLM(const ModelData& config, const Dict& labels, bool diagonal)
    : NLM(config, labels, diagonal) {}

  FactoredOutputNLM(const ModelData& config, const Dict& labels, bool diagonal, 
                                 const std::vector<int>& classes);

  Eigen::Block<WordVectorsType> class_R(const int c) {
    int c_start = indexes.at(c), c_end = indexes.at(c+1);
    return R.block(c_start, 0, c_end-c_start, R.cols());
  }

  const Eigen::Block<const WordVectorsType> class_R(const int c) const {
    int c_start = indexes.at(c), c_end = indexes.at(c+1);
    return R.block(c_start, 0, c_end-c_start, R.cols());
  }

  Eigen::VectorBlock<WeightsType> class_B(const int c) {
    int c_start = indexes.at(c), c_end = indexes.at(c+1);
    return B.segment(c_start, c_end-c_start);
  }

  const Eigen::VectorBlock<const WeightsType> class_B(const int c) const {
    int c_start = indexes.at(c), c_end = indexes.at(c+1);
    return B.segment(c_start, c_end-c_start);
  }

  int get_class(const WordId& w) const {
    assert(w >= 0 && w < int(word_to_class.size()) 
           && "ERROR: Failed to find word in class dictionary.");
    return word_to_class[w];
  }

  virtual Real l2_gradient_update(Real sigma) { 
    F -= F*sigma;
    FB -= FB*sigma;
    return NLM::l2_gradient_update(sigma) + F.array().square().sum() + FB.array().square().sum();
  }

  void reclass(std::vector<WordId>& train, std::vector<WordId>& test);

  virtual Real
  log_prob(const WordId w, const std::vector<WordId>& context, bool non_linear=false, bool cache=false) const {
    VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
    int width = config.ngram_order-1;
    int gap = width-context.size();
    assert(static_cast<int>(context.size()) <= width);
    for (int i=gap; i < width; i++)
      if (m_diagonal) prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
      else            prediction_vector += Q.row(context.at(i-gap)) * C.at(i);

    int c = get_class(w);

    // a simple non-linearity
    if (non_linear)
      prediction_vector = sigmoid(prediction_vector);

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

  void clear_cache() { 
    m_context_cache.clear(); 
    m_context_cache.reserve(1000000);
    m_context_class_cache.clear(); 
    m_context_class_cache.reserve(1000000);
  }

  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    NLM::save(ar, version);

    ar << word_to_class;
    ar << indexes;

    int F_rows=F.rows(), F_cols=F.cols();
    ar << F_rows << F_cols;
    ar << boost::serialization::make_array(F.data(), F_rows*F_cols);

    int FB_len=FB.rows();
    ar << FB_len;
    ar << boost::serialization::make_array(FB.data(), FB_len);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    NLM::load(ar, version);
    ar >> word_to_class;
    ar >> indexes;

    int F_rows=0, F_cols=0;
    ar >> F_rows >> F_cols;
    F = MatrixReal(F_rows, F_cols);
    ar >> boost::serialization::make_array(F.data(), F_rows*F_cols);

    int FB_len=0;
    ar >> FB_len;
    FB = VectorReal(FB_len);
    ar >> boost::serialization::make_array(FB.data(), FB_len);
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

public:
  std::vector<int> word_to_class;
  std::vector<int> indexes;
  MatrixReal F;
  VectorReal FB;

protected:
  mutable std::unordered_map<std::pair<int,Words>, Real> m_context_class_cache;
  mutable std::unordered_map<Words, Real, container_hash<Words> > m_context_cache;
};

class FactoredMaxentNLM : public FactoredOutputNLM {
 public:
  FactoredMaxentNLM();

  FactoredMaxentNLM(const ModelData& config, const Dict& labels, bool diagonal, 
                    const std::vector<int>& classes);

  virtual Real log_prob(
      WordId w, const std::vector<WordId>& context,
      bool nonlinear, bool cache) const;

  virtual Real l2_gradient_update(Real lambda);

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    FactoredOutputNLM::save(ar, version);
    ar << U << V;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    FactoredOutputNLM::load(ar, version);
    ar >> U >> V;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  UnconstrainedFeatureStore U;
  vector<UnconstrainedFeatureStore> V;
};

}

#endif // _NLM_H_
