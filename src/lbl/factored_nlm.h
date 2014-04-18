#pragma once

#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/nlm.h"
#include "lbl/word_to_class_index.h"

using namespace std;

namespace oxlm {

class FactoredNLM: public NLM {
 public:
  FactoredNLM();

  FactoredNLM(const ModelData& config, const Dict& labels);

  FactoredNLM(
      const ModelData& config, const Dict& labels,
      const boost::shared_ptr<WordToClassIndex>& index);

  Eigen::Block<WordVectorsType> class_R(const int c);

  const Eigen::Block<const WordVectorsType> class_R(const int c) const;

  Eigen::VectorBlock<WeightsType> class_B(const int c);

  const Eigen::VectorBlock<const WeightsType> class_B(const int c) const;

  int get_class(const WordId& w) const;

  virtual void l2GradientUpdate(Real minibatch_factor);

  virtual Real l2Objective(Real minibatch_factor) const;

  void reclass(
      const boost::shared_ptr<Corpus>& training_corpus,
      const boost::shared_ptr<Corpus>& test_corpus);

  virtual Real log_prob(
      const WordId w, const vector<WordId>& context,
      bool non_linear = false, bool cache = false) const;

  void clear_cache();

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar << boost::serialization::base_object<const NLM>(*this);

    ar << index;

    int F_rows=F.rows(), F_cols=F.cols();
    ar << F_rows << F_cols;
    ar << boost::serialization::make_array(F.data(), F_rows*F_cols);

    int FB_len=FB.rows();
    ar << FB_len;
    ar << boost::serialization::make_array(FB.data(), FB_len);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> boost::serialization::base_object<NLM>(*this);

    ar >> index;

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
  boost::shared_ptr<WordToClassIndex> index;
  MatrixReal F;
  VectorReal FB;

 protected:
  mutable unordered_map<pair<int,Words>, Real> m_context_class_cache;
  mutable unordered_map<Words, Real, container_hash<Words> > m_context_cache;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::FactoredNLM)
