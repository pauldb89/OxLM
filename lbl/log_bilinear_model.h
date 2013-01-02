#ifndef _LOG_BILINEAR_MODEL_H_
#define _LOG_BILINEAR_MODEL_H_

#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include "corpus/corpus.h"
#include "lbl/config.h"

namespace oxlm {

typedef float Real;
typedef std::vector<Real> Reals;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef boost::shared_ptr<MatrixReal>                       MatrixRealPtr;
typedef boost::shared_ptr<VectorReal>                       VectorRealPtr;

class LogBiLinearModel {
public:
  typedef Eigen::Map<MatrixReal> ContextTransformType;
  typedef std::vector<ContextTransformType> ContextTransformsType;
  typedef Eigen::Map<MatrixReal> WordVectorsType;
  typedef Eigen::Map<VectorReal> WeightsType;

public:
  LogBiLinearModel(const ModelData& config, const Dict& labels);
  LogBiLinearModel(const LogBiLinearModel& model);

  ~LogBiLinearModel() { delete [] m_data; }

  int labels() const { return m_labels.size(); }
  const Dict& label_set() const { return m_labels; }
  Dict& label_set() { return m_labels; }

  void l2_gradient_update(Real sigma) { W -= W*sigma; }

  void addModel(const LogBiLinearModel& model) { W += model.W; };

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
    ar << boost::serialization::make_array(m_data, m_data_size);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> config;
    ar >> m_labels;
    delete [] m_data;
    init(config, m_labels, false);
    ar >> boost::serialization::make_array(m_data, m_data_size);
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

public:
  ModelData config;

  ContextTransformsType C;
  WordVectorsType       R;
  WordVectorsType       Q;
  WeightsType           B;
  WeightsType           W;

private:
  void init(const ModelData& config, const Dict& labels, bool init_weights=false);

  Dict m_labels;

  int m_data_size;
  Real* m_data;
};

}

#endif // _LOG_BILINEAR_MODEL_H_
