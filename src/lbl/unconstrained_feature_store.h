#pragma once

#include <unordered_map>

#include "lbl/feature_context.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class UnconstrainedFeatureStore {
 public:
  UnconstrainedFeatureStore();

  UnconstrainedFeatureStore(int vector_size);

  VectorReal get(const vector<FeatureContext>& feature_contexts) const;

  void update(
      const vector<FeatureContext>& feature_contexts,
      const VectorReal& values);

  Real updateRegularizer(Real lambda);

  void update(const UnconstrainedFeatureStore& store);

  void updateSquared(const UnconstrainedFeatureStore& store);

  void updateAdaGrad(
      const UnconstrainedFeatureStore& gradient_store,
      const UnconstrainedFeatureStore& adagrad_store,
      Real step_size);

  void clear();

  size_t size() const;

  bool operator==(const UnconstrainedFeatureStore& store) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << vectorSize;

    size_t num_entries = featureWeights.size();
    ar << num_entries;
    for (const auto& entry: featureWeights) {
      ar << entry.first;
      const VectorReal weights = entry.second;
      ar << boost::serialization::make_array(weights.data(), weights.rows());
    }
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> vectorSize;

    size_t num_entries;
    ar >> num_entries;
    for (size_t i = 0; i < num_entries; ++i) {
      FeatureContext feature_context;
      ar >> feature_context;

      VectorReal weights = VectorReal::Zero(vectorSize);
      ar >> boost::serialization::make_array(weights.data(), vectorSize);

      featureWeights.insert(make_pair(feature_context, weights));
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 private:
  void update(const FeatureContext& feature_context, const VectorReal& values);

  unordered_map<FeatureContext, VectorReal, hash<FeatureContext>>
      featureWeights;
  int vectorSize;
};

} // namespace oxlm
