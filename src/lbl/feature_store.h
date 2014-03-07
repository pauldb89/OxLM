#pragma once

#include <unordered_map>

#include "lbl/feature.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class UnconstrainedFeatureStore {
 public:
  UnconstrainedFeatureStore();

  UnconstrainedFeatureStore(int vector_size);

  VectorReal get(const vector<Feature>& features) const;

  void update(const vector<Feature>& features, const VectorReal& values);

  Real updateRegularizer(Real lambda);

  void update(const UnconstrainedFeatureStore& store);

  void updateSquared(const UnconstrainedFeatureStore& store);

  void updateAdaGrad(
      const UnconstrainedFeatureStore& gradient_store,
      const UnconstrainedFeatureStore& adagrad_store,
      Real step_size);

  void clear();

  size_t size() const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << vector_size;

    size_t num_entries = feature_weights.size();
    ar << num_entries;
    for (const auto& entry: feature_weights) {
      ar << entry.first;
      const VectorReal weights = entry.second;
      ar << boost::serialization::make_array(weights.data(), weights.rows());
    }
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> vector_size;

    int num_entries;
    ar >> num_entries;
    for (int i = 0; i < num_entries; ++i) {
      Feature feature;
      ar >> feature;

      VectorReal weights;
      ar >> boost::serialization::make_array(weights.data(), vector_size);

      feature_weights.insert(make_pair(feature, weights));
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 private:
  void update(const Feature& feature, const VectorReal& values);

  unordered_map<Feature, VectorReal, hash<Feature>> feature_weights;
  int vector_size;
};

} // namespace oxlm
