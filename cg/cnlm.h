#ifndef _NLM_H_
#define _NLM_H_

#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Dense>

#include "corpus/corpus.h"
#include "cg/config.h"
#include "cg/utils.h"


namespace oxlm {


class ConditionalNLM {
public:
  typedef Eigen::Map<MatrixReal> ContextTransformType;
  typedef std::vector<ContextTransformType> ContextTransformsType;
  typedef Eigen::Map<MatrixReal> WordVectorsType;
  typedef Eigen::Map<VectorReal> WeightsType;

public:
  ConditionalNLM(const ModelData& config, const Dict& source_vocab, const Dict& target_vocab, const std::vector<int>& classes);
  ~ConditionalNLM() { delete [] m_data; }

  int source_types() const { return m_source_labels.size(); }
  int output_types() const { return m_target_labels.size(); }
  int context_types() const { return m_target_labels.size(); }

  int labels() const { return m_target_labels.size(); }
  const Dict& label_set() const { return m_target_labels; }
  Dict& label_set() { return m_target_labels; }

  Real l2_gradient_update(Real sigma) { 
    W -= W*sigma; 
    return W.array().square().sum();
  }

  WordId label_id(const Word& l) const { return m_target_labels.Lookup(l); }

  const Word& label_str(WordId i) const { return m_target_labels.Convert(i); }

  int num_weights() const { return m_data_size; }

  Real* data() { return m_data; }

  Real gradient(const std::vector<Sentence>& source_corpus, const std::vector<Sentence>& target_corpus, 
                const TrainingInstances &training_instances, Real lambda, WeightsType& g_W);

  Real log_prob(const WordId w, const std::vector<WordId>& context, const Sentence& source, bool cache=false) const;

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

  void clear_cache() { 
    m_context_cache.clear(); 
    m_context_cache.reserve(1000000);
    m_context_class_cache.clear(); 
    m_context_class_cache.reserve(1000000);
  }

  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar << config;
    ar << m_target_labels;
    ar << boost::serialization::make_array(m_data, m_data_size);

    ar << word_to_class;
    ar << indexes;
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> config;
    ar >> m_target_labels;
    delete [] m_data;
    init(false);
    ar >> boost::serialization::make_array(m_data, m_data_size);

    ar >> word_to_class;
    ar >> indexes;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  MatrixReal context_product(int i, const MatrixReal& v, bool transpose=false) const {
    if (config.diagonal)
      return (C.at(i).asDiagonal() * v.transpose()).transpose();
    else if (transpose) return v * C.at(i).transpose();
    else                return v * C.at(i);
  }

  void context_gradient_update(ContextTransformType& g_C, const MatrixReal& v,const MatrixReal& w) const {
    if (config.diagonal) g_C += (v.cwiseProduct(w).colwise().sum()).transpose();
    else                 g_C += (v.transpose() * w); 
  }

public:
  ModelData config;

  ContextTransformsType C;  // Context position transforms
  WordVectorsType       R;  // output word representations
  WordVectorsType       Q;  // context word representations
  WordVectorsType       F;  // class representations
  WordVectorsType       S;  // source word representations
  WeightsType           B;  // output word biases
  WeightsType           FB; // output class biases

  WeightsType           W;  // All the parameters in one vector

private:
  void init(bool init_weights=false);
  void allocate_data();
  void map_parameters(WeightsType& w, WordVectorsType& r, WordVectorsType& q, WordVectorsType& f, 
                      WordVectorsType& s, ContextTransformsType& c, WeightsType& b, WeightsType& fb) const;

  Dict m_source_labels, m_target_labels;
  int m_data_size;
  Real* m_data;

  std::vector<int> word_to_class; // map from word id to class
  std::vector<int> indexes;       // vocab spans for each class

  mutable std::unordered_map<std::pair<int,Words>, Real> m_context_class_cache;
  mutable std::unordered_map<Words, Real, container_hash<Words> > m_context_cache;
};

typedef std::shared_ptr<ConditionalNLM> ConditionalNLMPtr;

}
#endif // _NLM_H_
