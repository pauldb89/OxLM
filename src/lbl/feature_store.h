#pragma once

#include <unordered_map>

#include <Eigen/Sparse>

#include "feature.h"

using namespace std;

namespace oxlm {

// TODO(paul): Seriously, we should avoid redefining the same things everywhere.
typedef float Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1> ArrayReal;

class UnconstrainedFeatureStore {
 public:
  UnconstrainedFeatureStore(int vector_size);

  VectorReal get(const vector<Feature>& features) const;

  void update(const vector<Feature>& features, const VectorReal& values);

  void update(const UnconstrainedFeatureStore& store);

  void updateSquared(const UnconstrainedFeatureStore& store);

  void updateAdaGrad(
      const UnconstrainedFeatureStore& gradient_store,
      const UnconstrainedFeatureStore& adagrad_store,
      Real step_size);

  void clear();

  size_t size() const;

 private:
  void update(const Feature& feature, const VectorReal& values);

  unordered_map<Feature, VectorReal, hash<Feature>> feature_weights;
  int vector_size;
};

} // namespace oxlm
