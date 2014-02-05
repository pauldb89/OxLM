#ifndef CG_CNLM_H
#define CG_CNLM_H

#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <functional>
#include <fstream>
#include <vector>

#include <Eigen/Dense>

#include "corpus/corpus.h"
#include "cg/config.h"
#include "cg/utils.h"


//forward declaration
void gradient_check(const boost::program_options::variables_map& vm, oxlm::ModelData& config, const oxlm::Real epsilon);

namespace oxlm {

/*
 * This class implemented a conditional neural language model.
 * Bla bla bla about what it actually does.
 * Possibly a link to further reading.
 */
class CNLMBase {
public:
  typedef Eigen::Map<MatrixReal> ContextTransformType;
  typedef std::vector<ContextTransformType> ContextTransformsType;
  typedef Eigen::Map<MatrixReal> WordVectorsType;
  typedef Eigen::Map<VectorReal> WeightsType;

public:
  CNLMBase();
  CNLMBase(const ModelData& config, const Dict& target_vocab,
           const std::vector<int>& classes);

  CNLMBase(const ModelData& config, const Dict& source_vocab,
      const Dict& target_vocab, const std::vector<int>& classes);

  ~CNLMBase() { delete [] m_data; }
  void initWordToClass();

  void reinitialize(const ModelData& config, const Dict& source_vocab,
                    const Dict& target_vocab, const std::vector<int>& classes);
  void expandSource(const Dict& source_labels);

  int output_types() const { return m_target_labels.size(); }
  int context_types() const { return m_target_labels.size(); }
  int source_types() const { return m_source_labels.size(); }

  int labels() const { return m_target_labels.size(); }
  const Dict& label_set() const { return m_target_labels; }
  Dict& label_set() { return m_target_labels; }
  const Dict& source_label_set() const { return m_source_labels; }
  Dict& source_label_set() { return m_source_labels; }

  Real l2_gradient_update(Real sigma) {
    W -= W*sigma;
    return W.squaredNorm();
  }

  WordId label_id(const Word& l) const { return m_target_labels.Lookup(l); }

  const Word& label_str(WordId i) const { return m_target_labels.Convert(i); }

  virtual int num_weights() const { return m_data_size; }

  Real* data() { return m_data; }

  Real gradient_(
      const std::vector<Sentence>& target_corpus,
      const TrainingInstances& training_instances,
      // std::function<void(TrainingInstance, VectorReal)> source_repr_callback,
      // std::function<void(TrainingInstance, int, int, VectorReal)>
      //   source_grad_callback,
      Real l2, Real source_l2, Real*& g_ptr);

  void source_repr_callback(TrainingInstance t, int t_i,
                            VectorReal& r) = 0;
  void source_grad_callback(TrainingInstance t, int t_i,
                            int instance_counter,
                            const VectorReal& grads) = 0;

  void source_representation(const Sentence& source, int target_index,
                             VectorReal& result) const;
  void hidden_layer(const std::vector<WordId>& context,
                    const VectorReal& source, VectorReal& result) const;

  Real log_prob(const WordId w, const std::vector<WordId>& context,
                bool cache=false) const;
  Real log_prob(const WordId w, const std::vector<WordId>& context,
                const VectorReal& source, bool cache=false) const;
  void class_log_probs(const std::vector<WordId>& context,
                       const VectorReal& source,
                       const VectorReal& prediction_vector, VectorReal& result,
                       bool cache=false) const;
  void word_log_probs(int c, const std::vector<WordId>& context,
                      const VectorReal& source,
                      const VectorReal& prediction_vector, VectorReal& result,
                      bool cache=false) const;

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

  int map_class_to_word_index(int c, int wc) const {
    int c_start = indexes.at(c);
    return wc + c_start;
  }

  void clear_cache() {
    m_context_cache.clear();
    m_context_cache.reserve(1000000);
    m_context_class_cache.clear();
    m_context_class_cache.reserve(1000000);
  }

public:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar << config;
    ar << m_target_labels;
    ar << m_source_labels;
    ar << boost::serialization::make_array(m_data, m_data_size);

    ar << word_to_class;
    ar << indexes;
    ar << length_ratio;
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> config;
    ar >> m_target_labels;
    ar >> m_source_labels;
    delete [] m_data;
    // TODO(kmh): Clean up archiving
    init(false);
    ar >> boost::serialization::make_array(m_data, m_data_size);

    ar >> word_to_class;
    ar >> indexes;
    ar >> length_ratio;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();
};

  void map_parameters(Real*& ptr, WordVectorsType& r, WordVectorsType& q,
                      WordVectorsType& f, ContextTransformsType& c,
                      WeightsType& b, WeightsType& fb, WordVectorsType& s,
                      ContextTransformsType& t) const;


  MatrixReal context_product(int i, const MatrixReal& v,
                             bool transpose=false) const {
    if (config.diagonal)
      return (C.at(i).asDiagonal() * v.transpose()).transpose();
    else if (transpose) return v * C.at(i).transpose();
    else                return v * C.at(i);
  }

  void context_gradient_update(ContextTransformType& g_C, const MatrixReal& v,
                               const MatrixReal& w) const {
    if (config.diagonal) g_C += (v.cwiseProduct(w).colwise().sum()).transpose();
    else                 g_C += (v.transpose() * w);
  }

public:
  ModelData config;

  ContextTransformsType C;  // Context position transforms
  WordVectorsType       R;  // output word representations
  WordVectorsType       Q;  // context word representations
  WordVectorsType       F;  // class representations
  WeightsType           B;  // output word biases
  WeightsType           FB; // output class biases

  WeightsType           W;  // All the parameters in one vector
  Real length_ratio;

  friend void ::gradient_check(const boost::program_options::variables_map& vm, oxlm::ModelData& config, const oxlm::Real epsilon);
  // friend void ::gradient_check(const boost::program_options::variables_map& vm, oxlm::ModelData& cfg, oxlm::Real e);
  //friend void gradient_check(const variables_map& vm, ModelData& config, const Real epsilon);

protected:
  virtual void init(bool init_weights=false);
  virtual int calculateDataSize(bool allocate=false);

  Dict m_target_labels;
  int m_data_size;
  Real* m_data;

  std::vector<int> word_to_class; // map from word id to class
  std::vector<int> indexes;       // vocab spans for each class

  mutable std::unordered_map<std::pair<int,Words>, VectorReal>
    m_context_class_cache;
  mutable std::unordered_map<Words, VectorReal, container_hash<Words> >
    m_context_cache;
};

typedef std::shared_ptr<CNLMBase> CNLMBasePtr;

}  // namespace oxlm
#endif  // CG_CNLM_H
