#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "third_party/eigen/Eigen/Dense"
#include "third_party/eigen/Eigen/Sparse"
#include "third_party/smhasher/MurmurHash3.h"

using namespace std;
using namespace chrono;

namespace oxlm {

typedef float Real;

typedef int            WordId;
typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

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

inline VectorReal sigmoid(const VectorReal& v) {
  return (1.0 + (-v).array().exp()).inverse();
}

inline Array2DReal sigmoidDerivative(const MatrixReal& v) {
  return v.array() * (1 - v.array());
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
