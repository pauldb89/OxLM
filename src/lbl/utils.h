#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

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

// Helper functions for time measurement.

Time GetTime();

double GetDuration(const Time& start_time, const Time& stop_time);

} // namespace oxlm
