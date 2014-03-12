#pragma once

#include "lbl/utils.h"

#include <boost/serialization/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>

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

  Real z(const VectorReal& context) const;

  void train(const MatrixReal& contexts, const VectorReal& zs,
             Real step_size, int iterations, int approx_vectors);
  void train_lbfgs(const MatrixReal& contexts, const VectorReal& zs,
                   Real step_size, int iterations, int approx_vectors);

private:
  MatrixReal m_z_approx;
  VectorReal m_b_approx;
};

} // namespace oxlm
