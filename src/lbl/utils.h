#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "third_party/eigen/Eigen/Dense"
#include "third_party/eigen/Eigen/Sparse"
#include "third_party/smhasher/MurmurHash3.h"

#include "lbl/config.h"
#include "lbl/exceptions.h"
#include "lbl/operators.h"

using namespace std;
using namespace chrono;

namespace oxlm {

typedef float Real;

typedef int            WordId;
typedef vector<WordId> Sentence;

typedef vector<vector<int>>                        GlobalFeatureIndexes;
typedef boost::shared_ptr<GlobalFeatureIndexes>    GlobalFeatureIndexesPtr;
typedef unordered_map<int, vector<int>>            MinibatchFeatureIndexes;
typedef boost::shared_ptr<MinibatchFeatureIndexes> MinibatchFeatureIndexesPtr;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Map<VectorReal>                              VectorRealMap;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>  Array2DReal;
typedef Eigen::SparseVector<Real>                           SparseVectorReal;

typedef high_resolution_clock Clock;
typedef Clock::time_point     Time;


// Helper operations on vectors.

inline VectorReal softMax(const VectorReal& v) {
  Real max = v.maxCoeff();
  return (v.array() - (log((v.array() - max).exp().sum()) + max)).exp();
}

inline VectorReal logSoftMax(const VectorReal& v) {
  Real max = v.maxCoeff();
  Real log_z = log((v.array() - max).exp().sum()) + max;
  return v.array() - log_z;
}

inline VectorReal logSoftMax(const VectorReal& v, Real& log_z) {
  Real max = v.maxCoeff();
  log_z = log((v.array() - max).exp().sum()) + max;
  return v.array() - log_z;
}

template<class Matrix>
inline Matrix sigmoid(const Matrix& v) {
  return (1.0 + (-v).array().exp()).inverse().matrix();
}

inline Array2DReal sigmoidDerivative(const MatrixReal& v) {
  return v.array() * (1 - v.array());
}

template<class Matrix>
inline Matrix rectifier(const Matrix& v) {
  return v.unaryExpr(CwiseRectifierOp<Real>());
}

inline Array2DReal rectifierDerivative(const MatrixReal& v) {
  return v.unaryExpr(CwiseRectifierDerivativeOp<Real>());
}

template<class Matrix>
inline Matrix activation(
    const boost::shared_ptr<ModelData>& config, const Matrix& v) {
  switch (config->activation) {
    case IDENTITY:
      return v;
    case SIGMOID:
      return sigmoid(v);
    case RECTIFIER:
      return rectifier(v);
    default:
      throw UnknownActivationException();
  }
}

// Note: v here is the hidden layer after the activation has been applied.
// Be careful how you define future activations.
inline Array2DReal activationDerivative(
    const boost::shared_ptr<ModelData>& config, const MatrixReal& v) {
  switch (config->activation) {
    case IDENTITY:
      return v.array();
    case SIGMOID:
      return sigmoidDerivative(v);
    case RECTIFIER:
      return rectifierDerivative(v);
    default:
      throw UnknownActivationException();
  }
}

inline Real LogAdd(Real log_a, Real log_b) {
  if (log_a >= log_b) {
    return log_a + log1p(exp(log_b - log_a));
  } else {
    return log_b + log1p(exp(log_a - log_b));
  }
}

// Helper functions for time measurement.

Time GetTime();

double GetDuration(const Time& start_time, const Time& stop_time);

inline size_t MurmurHash(const vector<int>& data, int seed = 0) {
  size_t result[2] = {0, 0};
  MurmurHash3_x64_128(data.data(), data.size() * sizeof(int), seed, result);
  return result[0];
}

class NotImplementedException : public exception {
  virtual const char* what() const throw() {
    return "This method was not implemented";
  }
};

} // namespace oxlm
